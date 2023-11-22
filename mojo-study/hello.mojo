def your_function(a, b):
    ## let은 변하지 않는 변수를 선언할 때 사용.
    let c = a
    ## 아래 코드는 에러 발생
    ##  c = b

    if c != b:
        let d = b
        print(d)


def your_function2():

    let x:     Int = 42
    let y: Float64 = 17.0

    ## 변수를 선언할 때 값을 나중에 초기화 해 줄수도 있다.
    let z: Float32

    ## 파이썬처럼 컴프리헨션도 가능한 것 같다.
    z = 1.0 if x!= 0 else foo()

    print(z)


def foo() -> Float32: return 3.14


## 함수 정의할 때 def는 
## main 함수 안에서 코드를 작성하면 main 함수 호출하지 않아도
## mojo ~~~~~~.mojo로 실행하면 실행됨.
def main():

    print("hello mojo")
    for idx in range(9, 0, -3):
        print(idx)

    your_function(2, 3)
    your_function2()
