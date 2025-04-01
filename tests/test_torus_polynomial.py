import numpy as np
import sys
sys.path.append('/Users/mengxiangchen/Documents/Code/TFHE')
from tfhe.torus_polynomial import TorusPolynomial
from tfhe.poly import naive_polymul


def equal_torus_elem(t1, t2, atol=0.1, min=0, max=1):
    assert min <= t1 < max
    assert min <= t2 < max
    iatol = 1 - atol
    dist = abs(t1 - t2)
    if dist < atol or dist > iatol:
        return True
    else:
        return False


def test_torus_polynomial_int_encoding(log2_p):
    """测试多项式整数编码"""
    p = 2 ** log2_p
    print(f"测试多项式整数编码 (log2_p={log2_p})")
    for i in np.random.randint(0, p, size=10, dtype=np.uint64):
        u = TorusPolynomial.from_int(i, p)
        result = u.to_int(p)[0]
        assert result == i
    print(f"  ✅ 测试通过: 多项式整数编码 (log2_p={log2_p})")
    return True


def test_torus_polynomial_int_encoding_batched(log2_p, big_n):
    """测试批量多项式整数编码"""
    p = 2 ** log2_p
    print(f"测试批量多项式整数编码 (log2_p={log2_p}, big_n={big_n})")
    for i in np.random.randint(0, p, size=10, dtype=np.uint64):
        u = TorusPolynomial.from_int([i] * big_n, p, big_n)
        result = u.to_int(p)
        assert len(result) == big_n
        for j in range(big_n):
            assert result[j] == i
    print(f"  ✅ 测试通过: 批量多项式整数编码 (log2_p={log2_p}, big_n={big_n})")
    return True


def test_torus_polynomial_encoding(r, log2_p):
    """测试多项式实数编码"""
    p = 2 ** log2_p
    u = TorusPolynomial.from_real(r)
    result = u.to_real(p)[0]
    assert equal_torus_elem(result, r)
    return True


def test_torus_polynomial_encoding_batched(r, log2_p, big_n):
    """测试批量多项式实数编码"""
    p = 2 ** log2_p
    u = TorusPolynomial.from_real([r] * big_n, big_n)
    result = u.to_real(p)
    assert len(result) == big_n
    for i in range(big_n):
        assert equal_torus_elem(result[i], r)
    return True


def test_torus_polynomial_float_encoding(data_range, log2_p):
    """测试多项式浮点数编码"""
    p = 2 ** log2_p
    r = np.random.uniform(*data_range, size=(1)).item()
    u = TorusPolynomial.from_float(r, p, data_range)
    result = u.to_float(p, data_range)[0]
    precision = (data_range[1] - data_range[0]) / p
    assert np.allclose(result, r, atol=precision)
    return True


def test_torus_polynomial_float_encoding_batched(data_range, log2_p, big_n):
    """测试批量多项式浮点数编码"""
    p = 2 ** log2_p
    r = np.random.uniform(*data_range, size=(1)).item()
    u = TorusPolynomial.from_float([r] * big_n, p, data_range, big_n)
    result = u.to_float(p, data_range)
    precision = (data_range[1] - data_range[0]) / p
    assert len(result) == big_n
    for i in range(big_n):
        assert equal_torus_elem(
            result[i], r, atol=precision, min=data_range[0], max=data_range[1]
        )
    return True


def test_torus_polynomial_add_batched(r1, r2, big_n):
    """测试批量多项式加法"""
    p = 2 ** 16
    u1 = TorusPolynomial.from_real([r1] * big_n, big_n)
    u2 = TorusPolynomial.from_real([r2] * big_n, big_n)
    result = (u1 + u2).to_real(p)
    assert len(result) == big_n
    for i in range(big_n):
        assert equal_torus_elem(result[i], (r1 + r2) % 1)
    return True


def test_torus_polynomial_sub_batched(r1, r2, big_n):
    """测试批量多项式减法"""
    p = 2 ** 16
    u1 = TorusPolynomial.from_real([r1] * big_n, big_n)
    u2 = TorusPolynomial.from_real([r2] * big_n, big_n)
    result = (u1 - u2).to_real(p)
    assert len(result) == big_n
    for i in range(big_n):
        assert equal_torus_elem(result[i], (r1 - r2) % 1)
    return True

def test_torus_polynomial_mul_batched(log2_p, big_n):
    """测试多项式乘法"""
    p = 2 ** log2_p
    poly1 = np.random.random_integers(0, p - 1, size=big_n)
    poly2 = np.random.random_integers(0, p - 1, size=big_n)
    u1 = TorusPolynomial.from_int(poly1, p, big_n)
    u2 = TorusPolynomial.from_int(poly2, p, big_n)
    result = (u1 * u2).to_int(p)
    golden_result = naive_polymul(poly1, poly2, p)
    assert len(result) == big_n
    for i in range(big_n):
        assert equal_torus_elem(result[i], golden_result[i])
    return True

def run_tests():
    """运行所有测试用例"""
    total_tests = 0
    passed_tests = 0
    
    # 测试参数配置
    log2_p_values = [3, 5, 8, 16, 32]
    big_n_values = [2 ** 10, 2 ** 9, 2 ** 12]
    r_values = np.arange(0, 1, 0.1).tolist()
    data_ranges = [
        (0, 2),
        (-2, 1),
        (-5.5, -4),
        (-3.1, 3.5),
        (0.2, 1.4),
    ]
    log2_p_ext_values = [3, 5, 8, 16, 32, 63]

    print("\n=== 测试多项式整数编码 ===")
    for log2_p in log2_p_values:
        try:
            if test_torus_polynomial_int_encoding(log2_p):
                passed_tests += 1
            total_tests += 1
        except AssertionError as e:
            print(f"  ❌ 测试失败: 多项式整数编码 (log2_p={log2_p})")
            print(f"    错误: {e}")

    print("\n=== 测试批量多项式整数编码 ===")
    for log2_p in log2_p_values:
        for big_n in big_n_values:
            try:
                if test_torus_polynomial_int_encoding_batched(log2_p, big_n):
                    passed_tests += 1
                total_tests += 1
            except AssertionError as e:
                print(f"  ❌ 测试失败: 批量多项式整数编码 (log2_p={log2_p}, big_n={big_n})")
                print(f"    错误: {e}")

    print("\n=== 测试多项式实数编码 ===")
    for r in r_values:
        for log2_p in log2_p_ext_values:
            try:
                print(f"测试多项式实数编码 (r={r:.2f}, log2_p={log2_p})")
                if test_torus_polynomial_encoding(r, log2_p):
                    passed_tests += 1
                total_tests += 1
                print(f"  ✅ 测试通过: 多项式实数编码 (r={r:.2f}, log2_p={log2_p})")
            except AssertionError as e:
                print(f"  ❌ 测试失败: 多项式实数编码 (r={r:.2f}, log2_p={log2_p})")
                print(f"    错误: {e}")

    print("\n=== 测试批量多项式实数编码 ===")
    for r in r_values:
        for log2_p in log2_p_ext_values:
            for big_n in big_n_values:
                total_tests += 1
                try:
                    print(f"测试批量多项式实数编码 (r={r:.2f}, log2_p={log2_p}, big_n={big_n})")
                    if test_torus_polynomial_encoding_batched(r, log2_p, big_n):
                        passed_tests += 1                    
                    print(f"  ✅ 测试通过: 批量多项式实数编码 (r={r:.2f}, log2_p={log2_p}, big_n={big_n})")
                except AssertionError as e:
                    print(f"  ❌ 测试失败: 批量多项式实数编码 (r={r:.2f}, log2_p={log2_p}, big_n={big_n})")
                    print(f"    错误: {e}")

    print("\n=== 测试多项式浮点数编码 ===")
    for data_range in data_ranges:
        for log2_p in log2_p_ext_values:
            total_tests += 1
            try:
                print(f"测试多项式浮点数编码 (范围={data_range}, log2_p={log2_p})")
                if test_torus_polynomial_float_encoding(data_range, log2_p):
                    passed_tests += 1                
                print(f"  ✅ 测试通过: 多项式浮点数编码 (范围={data_range}, log2_p={log2_p})")
            except AssertionError as e:
                print(f"  ❌ 测试失败: 多项式浮点数编码 (范围={data_range}, log2_p={log2_p})")
                print(f"    错误: {e}")

    print("\n=== 测试批量多项式浮点数编码 ===")
    for data_range in data_ranges:
        for log2_p in log2_p_ext_values:
            for big_n in big_n_values:
                total_tests += 1
                try:
                    print(f"测试批量多项式浮点数编码 (范围={data_range}, log2_p={log2_p}, big_n={big_n})")
                    if test_torus_polynomial_float_encoding_batched(data_range, log2_p, big_n):
                        passed_tests += 1                    
                    print(f"  ✅ 测试通过: 批量多项式浮点数编码 (范围={data_range}, log2_p={log2_p}, big_n={big_n})")
                except AssertionError as e:
                    print(f"  ❌ 测试失败: 批量多项式浮点数编码 (范围={data_range}, log2_p={log2_p}, big_n={big_n})")
                    print(f"    错误: {e}")

    print("\n=== 测试批量多项式加法 ===")
    for r1 in r_values:
        for r2 in r_values:
            for big_n in big_n_values:
                total_tests += 1
                try:
                    print(f"测试批量多项式加法 (r1={r1:.2f}, r2={r2:.2f}, big_n={big_n})")
                    if test_torus_polynomial_add_batched(r1, r2, big_n):
                        passed_tests += 1                    
                    print(f"  ✅ 测试通过: 批量多项式加法 (r1={r1:.2f}, r2={r2:.2f}, big_n={big_n})")
                except AssertionError as e:
                    print(f"  ❌ 测试失败: 批量多项式加法 (r1={r1:.2f}, r2={r2:.2f}, big_n={big_n})")
                    print(f"    错误: {e}")

    print("\n=== 测试批量多项式减法 ===")
    for r1 in r_values:
        for r2 in r_values:
            for big_n in big_n_values:
                total_tests += 1
                try:
                    print(f"测试批量多项式减法 (r1={r1:.2f}, r2={r2:.2f}, big_n={big_n})")
                    if test_torus_polynomial_sub_batched(r1, r2, big_n):
                        passed_tests += 1
                    print(f"  ✅ 测试通过: 批量多项式减法 (r1={r1:.2f}, r2={r2:.2f}, big_n={big_n})")
                except AssertionError as e:
                    print(f"  ❌ 测试失败: 批量多项式减法 (r1={r1:.2f}, r2={r2:.2f}, big_n={big_n})")
                    print(f"    错误: {e}")
                    
    print("\n=== 测试多项式乘法 ===")
    for log2_p in log2_p_values:
        for big_n in big_n_values:
            total_tests += 1
            try:
                print(f"测试多项式乘法 (log2_p={log2_p}, big_n={big_n})")
                if test_torus_polynomial_mul_batched(log2_p, big_n):
                    passed_tests += 1
                print(f"  ✅ 测试通过: 多项式乘法 (log2_p={log2_p}, big_n={big_n})")
            except AssertionError as e:
                print(f"  ❌ 测试失败: 多项式乘法 (log2_p={log2_p}, big_n={big_n})")
                print(f"    错误: {e}")

    # 打印测试摘要
    print("\n=== 测试摘要 ===")
    print(f"总计测试: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    # 如果有测试失败，返回非零退出代码
    import sys
    sys.exit(0 if success else 1)
