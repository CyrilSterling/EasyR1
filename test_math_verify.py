from math_verify import parse, verify

# Test with simple expressions
gold1 = parse('x^2 + 2x + 1')
answer1 = parse('(x+1)^2')
result1 = verify(gold1, answer1)
print(f'Test 1 - Are x^2 + 2x + 1 and (x+1)^2 equivalent? {result1}')

# Test with LaTeX format for the same expressions
gold1b = parse('$x^2 + 2x + 1$')
answer1b = parse('$(x+1)^2$')
result1b = verify(gold1b, answer1b)
print(f'Test 1b - Are x^2 + 2x + 1 and (x+1)^2 in LaTeX equivalent? {result1b}')

# Test with simple numbers
gold2 = parse('2+2')
answer2 = parse('4')
result2 = verify(gold2, answer2)
print(f'Test 2 - Are 2+2 and 4 equivalent? {result2}')

# Test with LaTeX notation
gold3 = parse('${1,3} \\cup {2,4}$')
answer3 = parse('${1,2,3,4}$')
result3 = verify(gold3, answer3)
print(f'Test 3 - Are {1,3} ∪ {2,4} and {1,2,3,4} equivalent? {result3}')

print("Tests completed") 