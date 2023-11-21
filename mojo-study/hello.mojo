def main():

    print("hello world")
    for x in range(9, 0, -3): print(x)

## mojo style
def your_function(a, b):

    ## let은 값 변환이 안됨.
    let c = a

    if c!= b:
        let d = b
        print(d)

your_function(a = 2, b = 3)
