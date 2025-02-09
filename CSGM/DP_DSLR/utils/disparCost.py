"""
此模块提供了一系列辅助函数，主要用于：
  1. 计算左右图像之间的成本体（SAD 聚合）
  2. 利用成本值进行子像素二次拟合，得到拟合参数 a, b, c
  3. 若采用 ENCC 插值，则进行特殊的子像素处理
  4. 对二次拟合参数进行后处理（如边缘平滑、低信任区域抑制等）
  5. 实现 8 方向成本传播（简单使用均值滤波替代复杂 DP）
"""
from numba import njit

import numpy as np
import cv2

def compute_cost_volume(im_L, im_R, disp_candidates, window_size, sigma):
    """
    计算左右图像间的成本体，利用 SAD 在给定窗口上汇聚。
    
    改为使用高斯滤波进行聚合，模拟 MATLAB 中 imgaussfilt 的效果。
    
    输入:
      im_L, im_R: 灰度图像，格式 np.float64 (取值范围 [0,255])
      disp_candidates: 待测试的视差候选值数组（例如 np.arange(-6, 3)）
      window_size: 高斯窗口尺寸（建议与 encc_window 设置一致）
      sigma: 高斯核的标准差，推荐使用 params['gaussKerSigma'] 的值
      
    输出:
      cost_volume: 形状 (H, W, num_disp) 的成本体数组
    """
    H, W = im_L.shape
    num_disp = len(disp_candidates)
    cost_volume = np.zeros((H, W, num_disp), dtype=np.float64)
    
    for i, d in enumerate(disp_candidates):
        # 对右图进行水平平移，注意 d 可能为负值
        M = np.float32([[1, 0, d], [0, 1, 0]])
        shifted_R = cv2.warpAffine(im_R, M, (W, H),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # 修改为 borderValue=0
        diff = np.abs(im_L - shifted_R)
        # 采用高斯滤波进行聚合，模拟 MATLAB 的 imgaussfilt(-abs(...))
        cost = cv2.GaussianBlur(diff, (window_size, window_size), sigma)
        cost_volume[:, :, i] = cost
        
    return cost_volume


def subpixel_interpolate_disparity(cost_neig, disp_candidates, defval=1e-4):
    """
    对成本体进行子像素级的视差插值处理，与 MATLAB 版本类似：
      1. 对成本体 (cost_neig) 对应每个像素，取成本最小处和其左右邻域的成本值；
      2. 利用二次抛物线拟合求出子像素偏移量，例如：
          d_offset = (cost_left - cost_right) / (2 * (cost_left - 2 * cost_center + cost_right) + eps)
      3. 将该偏移量加到离散视差上，得到精细的子像素视差。
    
    参数:
      cost_neig: [H, W, D] 的成本体矩阵
      disp_candidates: 视差候选值列表或数组，形状为 (D,)
      defval: 用于防止除零错误的小值
      
    返回:
      dispar_int_val: 子像素级视差估计，形状为 [H, W]
    """
    import numpy as np
    eps = 1e-10
    H, W, D = cost_neig.shape
    # 选取最佳离散视差索引
    best_idx = np.argmin(cost_neig, axis=2)
    dispar_int_val = np.copy(disp_candidates[best_idx])
    
    # 只对中间部分做子像素插值，边界位置不做处理
    valid = (best_idx > 0) & (best_idx < D - 1)
    idx0, idx1 = np.where(valid)
    
    d_best = best_idx[idx0, idx1]
    cost_left = cost_neig[idx0, idx1, d_best - 1]
    cost_center = cost_neig[idx0, idx1, d_best]
    cost_right = cost_neig[idx0, idx1, d_best + 1]
    
    # 可以选择使用抛物线插值公式，也可以根据第二好候选的比例做调整，
    # 这里采用常用的抛物线公式：
    d_offset = (cost_left - cost_right) / (2 * (cost_left - 2 * cost_center + cost_right) + eps)
    # 注意：根据代价函数定义，有时插值符号可能需要取反，视实际情况而定。
    
    dispar_int_val[idx0, idx1] = disp_candidates[d_best] + d_offset
    return dispar_int_val

def disparCost(im_L, im_R, params):
    """
    根据左右图像及参数计算成本体、置信度及离散视差。

    输入:
      im_L, im_R: 灰度图（np.float64）
      params: 参数字典，要求含字段：
                'dispar_vals'：待测试的视差候选值数组
                'encc_window'：窗口尺寸
                'gaussKerSigma': 高斯滤波器的标准差
    输出:
      cost_neig: 成本体（汇聚后）数组，形状 (H, W, num_disp)
      conf_score: 置信度得分数组 (H, W)
      dispar_int_val: 每像素对应的最佳离散视差（直接取候选值）
      conf_score_no_suprress: 暂时与 conf_score 相同（后续可用于非极大值抑制）
    """
    disp_candidates = params['dispar_vals']
    window_size = int(params.get('encc_window', 31))
    sigma = params.get('gaussKerSigma', 11)  # 新增：获取高斯核标准差
    cost_neig = compute_cost_volume(im_L, im_R, disp_candidates, window_size, sigma)
    
    H, W = im_L.shape
    # 得到最佳候选索引（SAD 最小值）
    best_idx = np.argmin(cost_neig, axis=2)
    best_cost = cost_neig[np.arange(H)[:, None], np.arange(W), best_idx]
    # dispar_int_val = disp_candidates[best_idx]
    dispar_int_val = subpixel_interpolate_disparity(cost_neig, disp_candidates, defval=1e-4)
    
    # 为计算置信度，取第二小值与最小值之差作为参考
    cost_copy = cost_neig.copy()
    cost_copy[np.arange(H)[:, None], np.arange(W), best_idx] = 1e10
    second_best = np.min(cost_copy, axis=2)
    conf_score = second_best - best_cost
    conf_score = np.maximum(conf_score, 1e-6)  # 防止过小
    
    conf_score_no_suprress = conf_score.copy()
    
    return cost_neig, conf_score, dispar_int_val, conf_score_no_suprress


def genParab(cost_neig, dispar_int_val, params):
    """
    利用成本体中离散视差对应的成本值，进行 3 点二次拟合，
    得到子像素精度估计所用的拟合参数 a, b, c，使得子像素视差 x = -b/(2*a).
    
    对于每个像素，假设在最佳候选点附近有三个点：
       f(-1) = cost(最佳索引-1),  f(0) = cost(最佳索引),  f(1) = cost(最佳索引+1)
    解得：
       a = (f(-1)+f(1)-2*f(0))/2, b = (f(1)-f(-1))/2, c = f(0)
    
    对边界处（最佳索引在 0 或最后）的像素，则不做子像素修正。
    
    输入:
      cost_neig: 成本体数组 (H, W, num_disp)
      dispar_int_val: 离散视差（最佳候选对应的视差值）
      params: 参数字典（其中已含候选视差数组）
    
    输出:
      parab: 字典，包含 'a', 'b', 'c'，皆为 (H, W) 数组
    """
    H, W, num_disp = cost_neig.shape
    best_idx = np.argmin(cost_neig, axis=2)
    
    parab_a = np.ones((H, W), dtype=np.float64)
    parab_b = np.zeros((H, W), dtype=np.float64)
    parab_c = np.zeros((H, W), dtype=np.float64)
    
    valid = (best_idx > 0) & (best_idx < num_disp - 1)
    i_coords, j_coords = np.where(valid)
    idx = best_idx[valid]
    
    c_left   = cost_neig[i_coords, j_coords, idx - 1]
    c_center = cost_neig[i_coords, j_coords, idx]
    c_right  = cost_neig[i_coords, j_coords, idx + 1]
    
    parab_a[valid] = (c_left + c_right - 2 * c_center) / 2.0
    parab_b[valid] = (c_right - c_left) / 2.0
    parab_c[valid] = c_center

    # 对于边界或无效点，保持默认 a=1, b=0, c=c_center
    invalid = ~valid
    if np.any(invalid):
        best_idx_invalid = best_idx[invalid]
        parab_c[invalid] = cost_neig[np.arange(H)[:, None], np.arange(W), best_idx][invalid]
    
    return {'a': parab_a, 'b': parab_b, 'c': parab_c}


def estSubPixENCC(cost_neig, dispar_int_val, im_L, im_R, params):
    """
    若使用 ENCC 插值方式，对成本体进行子像素级处理。
    本实现与 genParab 类似，但返回一个 invalid_mask，用于标记边界点，
    后续可对低信任区域给予特别处理。
    
    输出:
      parab: 字典，包含 'a', 'b', 'c' 各系数
      invalid_mask: 形状 (H, W) 的 uint8 数组，边界或无效点置 1
    """
    H, W, num_disp = cost_neig.shape
    best_idx = np.argmin(cost_neig, axis=2)
    invalid_mask = np.zeros((H, W), dtype=np.uint8)
    
    parab = {'a': np.ones((H, W), dtype=np.float64),
             'b': np.zeros((H, W), dtype=np.float64),
             'c': np.zeros((H, W), dtype=np.float64)}
    
    valid = (best_idx > 0) & (best_idx < num_disp - 1)
    i_coords, j_coords = np.where(valid)
    idx = best_idx[valid]
    
    c_left   = cost_neig[i_coords, j_coords, idx - 1]
    c_center = cost_neig[i_coords, j_coords, idx]
    c_right  = cost_neig[i_coords, j_coords, idx + 1]
    
    parab['a'][valid] = (c_left + c_right - 2 * c_center) / 2.0
    parab['b'][valid] = (c_right - c_left) / 2.0
    parab['c'][valid] = c_center
    
    invalid_mask[~valid] = 1
    return parab, invalid_mask


def refineParab(parab, params):
    """
    对子像素二次拟合参数进行后处理：对低信任区域或边缘处进行平滑滤波。
    本例中采用 3x3 中值滤波。
    """
    kernel_size = 3
    a_ref = cv2.medianBlur(parab['a'].astype(np.float32), kernel_size)
    b_ref = cv2.medianBlur(parab['b'].astype(np.float32), kernel_size)
    c_ref = cv2.medianBlur(parab['c'].astype(np.float32), kernel_size)
    
    parab['a'] = a_ref.astype(np.float64)
    parab['b'] = b_ref.astype(np.float64)
    parab['c'] = c_ref.astype(np.float64)
    return parab



"""
本模块实现了 MATLAB 版 propCSGM 的传播过程，
完整复现了水平、垂直及对角线（主对角线与次对角线）传播的算法，
取代了之前简单采用 3x3 均值滤波的近似实现。
"""
def compute_P_adaptive(Pedges, a, prevA, P1param):
    """
    计算 Padaptive：
      先计算 weightPrev = -prevA 并限制在 [0,1] 内，
      然后返回 Padaptive = P1param * Pedges * weightPrev.
    注意：将 P1param 强制转换为标量以避免广播错误。
    """
    scalar_P1param = float(P1param)
    weightPrev = np.clip(-prevA, 0, 1)
    return scalar_P1param * Pedges * weightPrev

def compute_expected_value(prevA, prevB):
    """
    计算期望值 Ep = -prevB / (2 * prevA + epsilon)
    """
    epsilon = 1e-10
    return -prevB / (2.0 * (prevA + epsilon))

def compute_propagated_parabola(a, b, c, P1, Ep):
    """
    按照 MATLAB 实现：
      aNew = a - P1;
      bNew = b + 2*P1*Ep;
      cNew = c - P1*Ep^2;
    """
    a_new = a - P1
    b_new = b + 2 * P1 * Ep
    c_new = c - P1 * (Ep ** 2)
    return a_new, b_new, c_new

# -------------------------
# Propagation along a line (水平或垂直)
# -------------------------

def propLine(im_guide, parab, params, dir_flag):
    """
    复现 MATLAB propLine:
      对导引图 im_guide 计算相邻像素之差（水平或垂直），
      分别进行正向和反向传播更新（依次调用 compute_P_adaptive, compute_expected_value, compute_propagated_parabola），
      最后返回两组传播结果的和。
    """
    P1param = params['P1param']
    sigmaEdges = params['sigmaEdges']
    
    if dir_flag == 'H':
        guide = im_guide   # 正向传播时，guide 与 parab 方向一致
        Sa = parab['a'].copy()
        Sb = parab['b'].copy()
        Sc = parab['c'].copy()
        orig_a = parab['a'].copy()
    elif dir_flag == 'V':
        # 注意：此时传入的 im_guide 已经是 rot_im_guide（即转置后的导引图），
        # 因此不需要再执行 .T 。
        guide = im_guide
        Sa = parab['a'].T.copy()
        Sb = parab['b'].T.copy()
        Sc = parab['c'].T.copy()
        orig_a = parab['a'].T.copy()
    else:
        raise ValueError("dir_flag 必须为 'H' 或 'V'")
    
    H, W = guide.shape[:2]
    
    # 计算正向差分
    if guide.ndim == 3:
        df = np.sum(guide[:, 1:, :] - guide[:, :-1, :], axis=2) / guide.shape[2]
    else:
        df = guide[:, 1:] - guide[:, :-1]
    Pedges = np.concatenate([np.ones((H, 1)), np.exp(- (df ** 2) / (sigmaEdges ** 2))], axis=1)
    
    # 正向传播（从左到右）
    for ii in range(1, W):
        prevA = Sa[:, ii - 1]
        prevB = Sb[:, ii - 1]
        cur_orig_a = orig_a[:, ii]
        P_adaptive = compute_P_adaptive(Pedges[:, ii], cur_orig_a, prevA, P1param)
        Ep = compute_expected_value(prevA, prevB)
        a_new, b_new, c_new = compute_propagated_parabola(Sa[:, ii], Sb[:, ii], Sc[:, ii], P_adaptive, Ep)
        Sa[:, ii] = a_new
        Sb[:, ii] = b_new
        Sc[:, ii] = c_new
    
    # 反向传播（从右到左）
    if guide.ndim == 3:
        df_rev = np.sum(guide[:, :-1, :] - guide[:, 1:, :], axis=2) / guide.shape[2]
    else:
        df_rev = guide[:, :-1] - guide[:, 1:]
    Pedges_rev = np.concatenate([np.exp(- (df_rev ** 2) / (sigmaEdges ** 2)), np.ones((H, 1))], axis=1)
    Sa_rl = Sa.copy()
    Sb_rl = Sb.copy()
    Sc_rl = Sc.copy()
    for ii in range(W - 2, -1, -1):
        prevA = Sa_rl[:, ii + 1]
        prevB = Sb_rl[:, ii + 1]
        cur_orig_a = orig_a[:, ii]
        P_adaptive = compute_P_adaptive(Pedges_rev[:, ii], cur_orig_a, prevA, P1param)
        Ep = compute_expected_value(prevA, prevB)
        a_new, b_new, c_new = compute_propagated_parabola(Sa_rl[:, ii], Sb_rl[:, ii], Sc_rl[:, ii], P_adaptive, Ep)
        Sa_rl[:, ii] = a_new
        Sb_rl[:, ii] = b_new
        Sc_rl[:, ii] = c_new
    
    if dir_flag == 'H':
        parab_prop = {'a': Sa + Sa_rl, 'b': Sb + Sb_rl, 'c': Sc + Sc_rl}
    else:  # 'V'
        parab_prop = {'a': (Sa + Sa_rl).T, 'b': (Sb + Sb_rl).T, 'c': (Sc + Sc_rl).T}
    
    return parab_prop

# -------------------------
# Propagation along diagonals
# -------------------------

@njit
def diagonal_propagation_tl_inner(At, Bt, Ct, Orig_a, Pedges, P1param, d):
    """
    辅助函数：对第 d 条对角线上进行更新
    At, Bt, Ct: 当前的拟合参数矩阵，形状 (H, W)
    Orig_a: 原始 a 矩阵（用于参考置信度），形状 (H, W)
    Pedges: 对角线边缘权重矩阵，形状 (H, W)
    P1param: 标量参数
    d: 对角线编号，满足 i+j = d （d 从 0 到 H+W-2）
    """
    H, W = At.shape
    r_start = max(0, d - (W - 1))
    r_end = min(d, H - 1)
    eps = 1e-10
    # 对角线上若只有一个元素，则无需更新
    if r_end - r_start < 1:
        return

    # 对于对角线上每个元素（除第一个之外），其“前驱”为 (i-1, j-1)
    for i in range(r_start + 1, r_end + 1):
        j = d - i
        if j < 1 or j >= W:
            continue
        prevA = At[i - 1, j - 1]
        prevB = Bt[i - 1, j - 1]
        # 当前像素原始的置信度 a（可用于判定低信任区）；这里 Orig_a 保存初始值
        cur_orig_a = Orig_a[i, j]
        # 当前像素的原始拟合参数
        cur_a = At[i, j]
        cur_b = Bt[i, j]
        cur_c = Ct[i, j]
        # 计算前一像素的贡献权重（剪裁到 [0,1]）
        weightPrev = -prevA
        if weightPrev < 0.0:
            weightPrev = 0.0
        elif weightPrev > 1.0:
            weightPrev = 1.0
        # 计算自适应系数
        P_adaptive = P1param * Pedges[i, j] * weightPrev
        # 计算期望值，此处 epsilon 避免除 0
        Ep = -prevB / (2.0 * (prevA + eps))
        # 传播更新：用抛物线公式更新 a, b, c
        a_new = cur_a - P_adaptive
        b_new = cur_b + 2.0 * P_adaptive * Ep
        c_new = cur_c - P_adaptive * (Ep * Ep)
        At[i, j] = a_new
        Bt[i, j] = b_new
        Ct[i, j] = c_new

@njit
def diagonal_propagation_tl_vectorized(Sa, Sb, Sc, Orig_a, Pedges, P1param):
    """
    利用对角线分组的方法进行主对角线（左上方向）传播
    循环仅遍历每条对角线（共 H+W-1 条），内部调用 njit 加速函数处理更新。
    返回更新后的 (At, Bt, Ct)。
    """
    H, W = Sa.shape
    At = Sa.copy()
    Bt = Sb.copy()
    Ct = Sc.copy()
    # 对角线编号 d 满足 i+j = d，d 从 0 到 H+W-2
    for d in range(1, H + W):
        diagonal_propagation_tl_inner(At, Bt, Ct, Orig_a, Pedges, P1param, d)
    return At, Bt, Ct

def diagonal_propagation_tl_main(Sa, Sb, Sc, Orig_a, Pedges, P1param):
    """
    外部调用接口，执行左上方向传播（主对角线传播）
    """
    return diagonal_propagation_tl_vectorized(Sa, Sb, Sc, Orig_a, Pedges, P1param)

def diagonal_propagation_bl(Sa, Sb, Sc, Orig_a, Pedges, P1param):
    """
    利用左右翻转方式实现左下方向传播
    1. 先将各参数矩阵上下翻转（flipud）
    2. 调用主对角线传播得到更新结果
    3. 翻转回原来的顺序
    """
    Sa_flip = np.flipud(Sa)
    Sb_flip = np.flipud(Sb)
    Sc_flip = np.flipud(Sc)
    Orig_flip = np.flipud(Orig_a)
    Pedges_flip = np.flipud(Pedges)
    A_flip_new, B_flip_new, C_flip_new = diagonal_propagation_tl_main(Sa_flip, Sb_flip, Sc_flip, Orig_flip, Pedges_flip, P1param)
    A_new = np.flipud(A_flip_new)
    B_new = np.flipud(B_flip_new)
    C_new = np.flipud(C_flip_new)
    return A_new, B_new, C_new

def propDiag(im_guide, parab, params, dir_flag):
    """
    完全复现 MATLAB 中的 propDiag 传播过程：
      - 根据导引图 im_guide 与当前二次拟合参数 parab，
      - 采用左右两个传播方向：主对角线传播（左上方向）与次对角线传播（左下方向），
      - 利用向量化方法（仅一个对角线循环）加速计算，
      - 最后将两者结果相加得到最终传播结果。
      
    参数：
      im_guide: 导引图，可为灰度图 (H, W) 或 RGB 图 (H, W, C)
      parab: 字典，包含 'a', 'b', 'c'，各为 (H, W) 数组
      params: 参数字典，其中必须包含 'P1param'（标量）和 'sigmaEdges'
      dir_flag: 'MD' 表示主对角线传播；'OD' 表示次对角线传播（需要左右翻转）
    
    返回：
      parab_prop: 字典，包含传播后的 'a', 'b', 'c'
    """
    # 若处理次对角线方向（OD），先左右翻转 im_guide 与拟合参数
    if dir_flag == 'OD':
        im_guide = np.fliplr(im_guide)
        Sa_orig = np.fliplr(parab['a'].copy())
        Sb_orig = np.fliplr(parab['b'].copy())
        Sc_orig = np.fliplr(parab['c'].copy())
        Orig_a = np.fliplr(parab['a'].copy())
    else:  # 'MD'
        Sa_orig = parab['a'].copy()
        Sb_orig = parab['b'].copy()
        Sc_orig = parab['c'].copy()
        Orig_a = parab['a'].copy()
    
    sigmaEdges = params['sigmaEdges']
    P1param = float(params['P1param'])
    H, W = Sa_orig.shape

    # 计算左上方向的边缘权重 PedgesTopLeft：
    #   对于每个像素 (i,j)（i>=1, j>=1），用相邻对角线像素差值计算边缘保留系数，
    #   MATLAB 中的实现为：exp( - (df^2)/(sigmaEdges^2) )；边界处置为1。
    if im_guide.ndim == 3:
        df = np.mean(im_guide[1:, 1:, :] - im_guide[:-1, :-1, :], axis=2)
    else:
        df = im_guide[1:, 1:] - im_guide[:-1, :-1]
    PedgesTopLeft = np.ones((H, W), dtype=Sa_orig.dtype)
    PedgesTopLeft[1:, 1:] = np.exp(- (df ** 2) / (sigmaEdges ** 2))
    
    # 左上方向传播（MD 或 OD 下均相同，后续 OD 会翻转回来）
    At_tl, Bt_tl, Ct_tl = diagonal_propagation_tl_main(Sa_orig, Sb_orig, Sc_orig, Orig_a, PedgesTopLeft, P1param)
    
    # 计算左下方向的边缘权重 PedgesBotLeft：
    #   采用相反方向的差分
    if im_guide.ndim == 3:
        df_bot = np.mean(im_guide[:-1, 1:, :] - im_guide[1:, :-1, :], axis=2)
    else:
        df_bot = im_guide[:-1, 1:] - im_guide[1:, :-1]
    PedgesBotLeft = np.ones((H, W), dtype=Sa_orig.dtype)
    PedgesBotLeft[:-1, 1:] = np.exp(- (df_bot ** 2) / (sigmaEdges ** 2))
    
    At_bl, Bt_bl, Ct_bl = diagonal_propagation_bl(Sa_orig, Sb_orig, Sc_orig, Orig_a, PedgesBotLeft, P1param)
    
    # 合并两种传播结果
    A_prop = At_tl + At_bl
    B_prop = Bt_tl + Bt_bl
    C_prop = Ct_tl + Ct_bl
    
    # 若初始为 OD（次对角线），翻转回原方向
    if dir_flag == 'OD':
        A_prop = np.fliplr(A_prop)
        B_prop = np.fliplr(B_prop)
        C_prop = np.fliplr(C_prop)
    
    return {'a': A_prop, 'b': B_prop, 'c': C_prop}

# -------------------------
# 完全复现 MATLAB 传播流程的主函数
# -------------------------

def propCSGM(parab, im_guide, params):
    """
    复现 MATLAB 中的 propCSGM:
      根据导引图计算水平、垂直和对角线方向的传播，
      对传播结果进行叠加，并根据迭代次数更新先验 parab，
      最终返回传播后的二次拟合参数 sum_parab.
    """
    # 保证 im_guide 与 parab['a'] 尺寸一致
    if im_guide.shape != parab['a'].shape:
        im_guide = im_guide.T

    # 对于垂直传播，构造转置后的导引图
    if im_guide.ndim == 3:
        rot_im_guide = np.transpose(im_guide, (1, 0, 2))
    else:
        rot_im_guide = im_guide.T

    aInit = np.ones_like(parab['a'])
    num_iter = params['numIter'][params['idx_pyr']]
    
    if num_iter > 0:
        sum_parab_a = np.zeros_like(parab['a'])
        sum_parab_b = np.zeros_like(parab['a'])
        sum_parab_c = np.zeros_like(parab['a'])
    else:
        sum_parab_a = parab['a']
        sum_parab_b = parab['b']
        sum_parab_c = parab['c']
    
    for i in range(num_iter):
        # 水平传播：传入原始 im_guide
        parab_h = propLine(im_guide, parab, params, 'H')
        # 垂直传播：传入 rot_im_guide（已转置）
        parab_v = propLine(rot_im_guide, parab, params, 'V')
        # 主对角线传播（propDiag）与次对角线传播（propDiag）的实现类似，这里略去，按 MATLAB 流程实现即可
        parab_MD = propDiag(im_guide, parab, params, 'MD')
        parab_OD = propDiag(im_guide, parab, params, 'OD')
        
        sum_parab_a = parab_h['a'] + parab_v['a'] + parab_MD['a'] + parab_OD['a']
        sum_parab_b = parab_h['b'] + parab_v['b'] + parab_MD['b'] + parab_OD['b']
        sum_parab_c = parab_h['c'] + parab_v['c'] + parab_MD['c'] + parab_OD['c']
        
        if i != num_iter - 1:
            numPaths = 8
            parab['a'] = (sum_parab_a / numPaths) * aInit
            parab['b'] = (sum_parab_b / numPaths) * aInit
            parab['c'] = (sum_parab_c / numPaths) * aInit
    
    return {'a': sum_parab_a, 'b': sum_parab_b, 'c': sum_parab_c}