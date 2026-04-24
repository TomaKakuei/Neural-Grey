import rawpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import sys

class VignetteAnalyzer:
    def __init__(self, raw_path):
        # 移除了固定的 resize_target 参数，改为在处理时自动决定
        self.raw_path = raw_path
        self.img_data = None
        self.optical_center = None 
        self.max_brightness = 0
        self.radial_profile = None 
        self.fallout_01ev_radius = None
        self.popt = None 

    def load_and_preprocess(self):
        """
        1. 读取 RAW
        2. 自动识别 2:3 或 3:4 画幅并设定目标尺寸
        3. Resize
        4. 噪点清洗 & 光心校准
        """
        print(f"[-] Loading RAW: {self.raw_path}")
        with rawpy.imread(self.raw_path) as raw:
            # postprocess 参数保持不变，保证线性度
            rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # === 新增：自动画幅识别与匹配逻辑 ===
        h, w = gray.shape
        ratio = max(w, h) / min(w, h) # 计算长边/短边比率
        
        # 定义目标尺寸 (W, H) - 注意 OpenCV resize 接受的是 (width, height)
        # 这里为了通用性，我们先判断横竖构图
        is_landscape = w > h
        
        target_size = None # (width, height)

        # 判定比率 (允许 0.05 的误差，防止 1.49999 这种浮点误差)
        if abs(ratio - 1.5) < 0.05:  
            # 匹配 3:2 画幅 -> 目标 3000 x 2000
            print("[-] Detected Aspect Ratio: 3:2")
            if is_landscape:
                target_size = (3000, 2000)
            else:
                target_size = (2000, 3000)
                
        elif abs(ratio - 1.333) < 0.05:
            # 匹配 4:3 画幅 -> 目标 4000 x 3000
            print("[-] Detected Aspect Ratio: 4:3")
            if is_landscape:
                target_size = (4000, 3000)
            else:
                target_size = (3000, 4000)
        
        else:
            # 兜底逻辑：如果是 16:9 或 1:1 等其他奇怪画幅
            # 默认将长边缩放到 4000，保持原比例
            print(f"[-] Detected Non-standard Aspect Ratio: {ratio:.2f}")
            scale = 4000 / max(w, h)
            target_size = (int(w * scale), int(h * scale))

        print(f"[-] Resizing from {w}x{h} to {target_size}")
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        # ============================================

        
        # 下面的代码完全不用动
        print("[-] Cleaning noise (outliers > 40%)...")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
        diff = np.abs(gray - blurred)
        mask = diff > (0.40 * blurred) 
        gray[mask] = blurred[mask] 
        
        self.img_data = gray

        # 光心校准
        heavy_blur = cv2.GaussianBlur(gray, (101, 101), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(heavy_blur)
        self.optical_center = maxLoc
        self.max_brightness = maxVal
        print(f"[-] Optical Center Found: {self.optical_center}, Max Brightness: {self.max_brightness}")
        

    def extract_radial_profile(self):
        """
        提取径向暗角数据，并检测 0.1EV Fallout 位置
        """
        h, w = self.img_data.shape
        cx, cy = self.optical_center
        
        # 生成坐标网格
        y, x = np.indices((h, w))
        
        # 计算每个像素到光心的距离 (r)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # 扁平化
        r_flat = r.ravel()
        intensity_flat = self.img_data.ravel()
        
        # 为了减少计算量，每隔 1.0 个像素半径进行一次采样 (Binning)
        r_int = r_flat.astype(int)
        tbin = np.bincount(r_int, intensity_flat)
        nr = np.bincount(r_int)
        radial_profile_y = tbin / nr # 平均亮度
        radial_profile_x = np.arange(len(radial_profile_y)) # 半径
        
        # 归一化亮度 (0.0 - 1.0)
        radial_profile_y = radial_profile_y / self.max_brightness
        
        self.radial_profile = (radial_profile_x, radial_profile_y)
        
        # 检测 0.1 EV Falloff
        # -0.1 EV = 2^(-0.1) ≈ 0.933
        threshold = 2 ** (-0.1)
        # 找到第一个低于阈值的索引
        indices = np.where(radial_profile_y < threshold)[0]
        if len(indices) > 0:
            self.fallout_01ev_radius = indices[0]
            print(f"[-] 0.1 EV Falloff detected at radius: {self.fallout_01ev_radius} px")
        else:
            # === 修改部分：直接终止程序 ===
            print("\n[!] CRITICAL: Vignetting is too light (< 0.1 EV falloff).")
            print("[!] No center filter needed. Process terminated to prevent invalid fabrication.")
            sys.exit(1) # 退出代码 1 表示非正常结束（中止），0 表示正常结束

    def kang_weiss_model(self, r, a, b, c):
        """
        标准暗角模型公式
        L(r) = L_max * (1 + a*r^2 + b*r^4 + c*r^6)
        注意：这里的 r 通常需要归一化到 [0,1] 或者 [0, max_diagonal] 避免数值溢出
        """
        # 为了拟合稳定，输入 r 建议归一化。
        # 但为了保留物理意义，我们会在外部控制 r 的单位
        return 1.0 * (1 + a * (r**2) + b * (r**4) + c * (r**6))

    def fit_model(self):
        """
        拟合 Kang-Weiss 模型
        """
        r_x, r_y = self.radial_profile
        
        # 为了数值稳定性，将像素半径归一化到 [0, 1] 之间进行拟合
        # max_r = np.max(r_x)
        # r_norm = r_x / max_r
        
        # 限制只拟合有效数据（比如去除边缘极度噪点，或者只拟合到图像边界）
        valid_idx = np.where(r_y > 0.05) # 去除死黑噪点
        r_fit = r_x[valid_idx]
        y_fit = r_y[valid_idx]

        # 初始猜测
        p0 = [-0.0001, 0.0, 0.0] 
        
        try:
            # 拟合
            self.popt, _ = curve_fit(self.kang_weiss_model, r_fit, y_fit, p0=p0)
            print(f"[-] Model Fitted: a={self.popt[0]:.2e}, b={self.popt[1]:.2e}, c={self.popt[2]:.2e}")
        except Exception as e:
            print(f"[!] Fitting failed: {e}")
            self.popt = [0, 0, 0]

    def geometric_mapping(self, focal_length_mm, sensor_diag_mm, filter_total_diameter_mm, front_element_dia_mm=None):
        """
        核心物理映射逻辑：
        Pixel -> Angle -> Physical Filter Radius
        """
        print("\n=== Geometric Mapping Report ===")
        
        # 1. 传感器最大视角 (Sensor Max Angle)
        # theta_max = arctan( sensor_half_diag / focal_length )
        sensor_half_diag = sensor_diag_mm / 2.0
        theta_max_rad = np.arctan(sensor_half_diag / focal_length_mm)
        theta_max_deg = np.degrees(theta_max_rad)
        print(f"Sensor Coverage Half-Angle: {theta_max_deg:.2f}°")

        # 2. 建立 像素半径(px) 到 物理角度(rad) 的映射
        # r_px_max 对应 theta_max
        # 假设也是 rectilinear 投影: r_px = k * tan(theta)
        # 所以 theta(r_px) = arctan( r_px * (tan(theta_max) / r_px_max) )
        r_px_data, intensity_data = self.radial_profile
        max_px_radius = r_px_data[-1]
        
        def px_to_theta(r_pixel):
            # 线性投影假设 (Rectilinear Lens)
            # scale_factor = tan(theta_max) / max_px_radius
            tan_theta = r_pixel * (np.tan(theta_max_rad) / max_px_radius)
            return np.arctan(tan_theta)

        # 3. 建立 物理角度(rad) 到 滤镜半径(mm) 的映射
        # 这里的映射取决于滤镜安装位置。
        # 假设滤镜紧贴前组，且光线以角度 theta 射入。
        # 简单几何模型：Filter_Radius = Focal_Length * tan(theta) ??? 
        # 不，对于广角镜头的中灰镜，更实用的工程近似是：
        # Filter_Radius(theta) 线性对应于前组镜片的物理孔径分布。
        # 但最通用的模型依然是 r_filter = Distance_to_Pupil * tan(theta).
        # 如果不知道 Pupil 距离，我们通常假设滤镜覆盖了整个视场。
        
        # ** Hack for "Custom Fit": **
        # 我们假设滤镜的有效直径 (filter_total_diameter) 对应镜头的某个最大设计视角。
        # 但为了严谨，我们直接输出 "基于角度的亮度公式"。
        # 并计算：当前的传感器数据，覆盖了滤镜直径的多少 mm？
        
        # 假设滤镜安装在光心前方 d 处。
        # 如果没有 d，我们无法得出绝对毫米值。
        # 但通常 d ≈ focal_length (对于大画幅广角结构). 
        # 或者更简单：我们定义滤镜半径就是 r = f * tan(theta) (理想针孔模型在底片平面的投影，反向投射到滤镜平面)
        
        # 让我们计算传感器边缘对应的“等效滤镜半径”
        r_filter_covered_mm = focal_length_mm * np.tan(theta_max_rad) 
        print(f"Data Coverage on Filter Plane (approx): Radius {r_filter_covered_mm:.2f} mm")
        
        filter_radius_max = filter_total_diameter_mm / 2.0
        print(f"Physical Filter Max Radius: {filter_radius_max:.2f} mm")
        
        coverage_percent = (r_filter_covered_mm / filter_radius_max) * 100
        print(f"Sensor Data covers {coverage_percent:.1f}% of the Filter Radius.")

        # 4. 生成最终输出数据
        # 我们生成一个从 0 到 filter_radius_max 的数组 (以 0.1mm 为步长)
        target_r_mm = np.arange(0, filter_radius_max, 0.1)
        
        # 反推每个 mm 对应的角度 theta
        # r_mm = f * tan(theta)  =>  theta = arctan(r_mm / f)
        target_theta = np.arctan(target_r_mm / focal_length_mm)
        
        # 再将 theta 转回 像素半径 (px) 以便代入我们的拟合公式
        # theta = arctan(C * r_px) => tan(theta) = C * r_px => r_px = tan(theta) / C
        # C = tan(theta_max_raw) / max_px_radius_raw
        C = np.tan(theta_max_rad) / max_px_radius
        target_r_px = np.tan(target_theta) / C
        
        # 5. 计算最终亮度曲线 (混合模式)
        # 在传感器覆盖范围内 (target_r_px < max_px_radius)，我们既有 raw 数据也有拟合数据
        # 在传感器覆盖范围外，我们只能用拟合数据 (Extrapolation)
        
        # 构建 Raw 插值函数
        raw_interpolator = interp1d(r_px_data, intensity_data, bounds_error=False, fill_value="extrapolate")
        
        final_curve_raw = raw_interpolator(target_r_px)
        final_curve_fit = self.kang_weiss_model(target_r_px, *self.popt)
        
        # 修正：Raw 插值在超出范围后会失效(变平或乱飞)，所以必须在 cutoff 处强制切换为 fit
        mask_outside = target_r_px > max_px_radius
        final_curve_hybrid = final_curve_raw.copy()
        final_curve_hybrid[mask_outside] = final_curve_fit[mask_outside]

        return target_r_mm, final_curve_hybrid, final_curve_fit

# === 使用示例 ===
if __name__ == "__main__":
    # 参数设置
    RAW_FILE = "test_vignette.ARW"  # 替换你的 raw 文件
    FOCAL_LEN = 47.0      # 47mm 镜头
    SENSOR_DIAG = 150.0   # 4x5 画幅对角线 (mm)
    FILTER_DIA = 77.0     # 滤镜直径 (mm)

    analyzer = VignetteAnalyzer(RAW_FILE)
    analyzer.load_and_preprocess()
    analyzer.extract_radial_profile()
    analyzer.fit_model()
    
    # 核心映射
    r_mm, curve_hybrid, curve_fit = analyzer.geometric_mapping(
        focal_length_mm=FOCAL_LEN, 
        sensor_diag_mm=SENSOR_DIAG, 
        filter_total_diameter_mm=FILTER_DIA
    )

    # 绘图验证
    plt.figure(figsize=(10, 6))
    plt.plot(r_mm, curve_hybrid, label="Hybrid Curve (Raw Data + Extrapolation)", color='blue', linewidth=2)
    plt.plot(r_mm, curve_fit, '--', label="Kang-Weiss Fit Model (Ideal Physics)", color='red', alpha=0.7)
    
    plt.axvline(x=(SENSOR_DIAG/2.0 * (FOCAL_LEN/FOCAL_LEN)), color='green', linestyle=':', label='Sensor Edge (Approx)') 
    # 注意：上面的 Sensor Edge 是一阶近似，实际取决于投影几何
    
    plt.title(f"Vignetting Profile mapped to {FILTER_DIA}mm Filter")
    plt.xlabel("Physical Radius on Filter (mm)")
    plt.ylabel("Light Transmission (Normalized)")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

    # 输出两个公式的数据点，供下一步制造机器使用
    # format: radius_mm, brightness_value
    output_data = np.column_stack((r_mm, curve_hybrid))
    # np.savetxt("filter_profile.csv", output_data, delimiter=",", header="radius_mm,brightness")