def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)

ans = factorial(6)

print(ans)

