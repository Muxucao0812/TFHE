import numpy as np
import sys
sys.path.append('/Users/mengxiangchen/Documents/Code/TFHE')
from tfhe.torus import Torus

def equal_torus_elem(t1, t2, atol=0.1):
    assert 0 <= t1 < 1
    assert 0 <= t2 < 1
    iatol = 1 - atol
    dist = abs(t1 - t2)
    if dist < atol or dist > iatol:
        return True
    else:
        return False

def test_torus_int_encoding(log2_p):
    """测试整数编码到环上的转换"""
    torus = Torus()
    p = 2 ** log2_p
    print(f"测试整数编码 (log2_p={log2_p})")
    for i in np.random.randint(0, p, size=10, dtype=np.uint64):
        u = torus.from_int(i, p)
        result = u.to_int(p)
        print(f"  结果: {result}, 输入: {i}")
        assert result == i
    print(f"  ✅ 测试通过: 整数编码 (log2_p={log2_p})")
    return True

def test_torus_encoding(r, log2_p):
    """测试实数编码到环上的转换"""
    torus = Torus()
    p = 2 ** log2_p
    u = torus.from_real(r)
    result = u.to_real(p)
    print(f"  结果: {result:.6f}, 输入: {r:.6f}")
    # 计算误差
    error = np.abs(result - r) / r
    print(f"  误差: {error:.6f}")
    assert equal_torus_elem(result, r)
    return True

def test_float_encoding(data_range, log2_p):
    """测试浮点数编码到环上的转换"""
    torus = Torus()
    p = 2 ** log2_p
    r = np.random.uniform(*data_range, size=(1)).item()
    u = torus.from_float(r, p, data_range)
    result = u.to_float(p, data_range)
    precision = (data_range[1] - data_range[0]) / p
    print(f"  结果: {result:.6f}, 输入: {r:.6f}")
    assert np.allclose(result, r, atol=precision)
    error = np.abs(result - r) / r
    print(f"  误差: {error:.6f}")
    return True

def run_tests():
    """运行所有测试用例"""
    total_tests = 0
    passed_tests = 0
    non_passed_tests = 0
    
    print("\n=== 测试整数编码 ===")
    log2_p_values = [3, 5, 8, 16, 32]
    for log2_p_item in log2_p_values:
        total_tests += 1
        try:
            if test_torus_int_encoding(log2_p_item):
                passed_tests += 1            
        except AssertionError as e:
            print(f"  ❌ 测试失败: 整数编码 (log2_p={log2_p_item})")
            print(f"    错误: {e}")
            non_passed_tests += 1
            
    print("\n=== 测试实数编码 ===")
    log2_p_values = [3, 5, 8, 16, 32]
    r_values = np.arange(0, 1, 0.1).tolist()
    for log2_p_item in log2_p_values:
        for r_item in r_values:
            total_tests += 1
            try:
                print(f"测试实数编码 (r={r_item:.2f}, log2_p={log2_p_item})")
                if test_torus_encoding(r_item, log2_p_item):
                    passed_tests += 1                
            except AssertionError as e:
                print(f"  ❌ 测试失败: 实数编码 (r={r_item:.2f}, log2_p={log2_p_item})")
                print(f"    错误: {e}")
                non_passed_tests += 1

    print("\n=== 测试浮点数编码 ===")
    data_ranges = [
        (0, 2),
        (-2, 1),
        (-5.5, -4),
        (-3.1, 3.5),
        (0.2, 1.4),
    ]
    log2_p_values = [3, 5, 8, 16, 32, 63]
    for data_range_item in data_ranges:
        for log2_p_item in log2_p_values:
            total_tests += 1
            try:
                print(f"测试浮点数编码 (范围={data_range_item}, log2_p={log2_p_item})")
                if test_float_encoding(data_range_item, log2_p_item):
                    passed_tests += 1
            except AssertionError as e:
                print(f"  ❌ 测试失败: 浮点数编码 (范围={data_range_item}, log2_p={log2_p_item})")
                print(f"    错误: {e}")
                non_passed_tests += 1
    
    # 打印测试摘要
    print("\n=== 测试摘要 ===")
    print(f"总计测试: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"未通过测试: {non_passed_tests}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_tests()
    # 如果有测试失败，返回非零退出代码
    import sys
    sys.exit(0 if success else 1)

