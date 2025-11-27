from trading_sim import SimpleFuzzy

fuzzy = SimpleFuzzy()

# Test Case yang gagal di simulasi
dist_val = 4.78
slope_val = -206.6

# 1. Test Trapezoid Logic Manual
# def trapezoid(x, a, b, c, d):
#    return max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0)

x = 4.78
a, b, c, d = -0.1, 0, 2.0, 5.0
term1 = (x - a) / (b - a)
term3 = (d - x) / (d - c)
res = max(min(term1, 1, term3), 0)

print(f"Manual Calc for Dist {x}: Term1={term1:.2f}, Term3={term3:.2f}, Result={res:.2f}")

# 2. Test Class Method
inputs = fuzzy.fuzzify(dist_val, slope_val)
print("Fuzzify Output:", inputs)

score = fuzzy.infer(inputs, -1) # Trend Down
print("Inference Score:", score)
