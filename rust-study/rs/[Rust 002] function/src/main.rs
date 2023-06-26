// 함수 선언에 fn를 사용하고, 중괄호 안에 로직을 작성한다.
// 파라미터의 자료형을 선언하고, 화살표를 이용해 리턴 값의 자료형을 표기한다.
fn add(num1: i32, num2: i32) -> i32 {
    return num1 + num2
}


// 러스트에서는 함수에서 값을 변환할 때 return을 생략 가능하다.
fn add_(num1: i32, num2: i32) -> i32 {
    num1 + num2
}


// 여러 개의 값을 반환할 때 화살표를 이용한 리턴 값의 자료혈 표기는 튜플을 사용해 표기한다.
fn swap(num1: i32, num2: i32) -> (i32, i32) {
    (num2, num1)
}


// 반환 값이 없는 경우에 화살표로 ()을 표기해 반환값이 없음을 명시해야한다.
fn do_nothing() -> () {
    return ();
}


fn me_too() {}


fn main() {
    println!("  add function return value : {}",   add(1, 2));
    println!(" add_ function return value : {}",  add_(1, 2));

    let (num1, num2) = swap(1, 2);
    println!("swap_ function return value : {num1} {num2}");

    println!("{:?}", do_nothing());
    println!("{:?}", me_too());

    // rust에도 파이썬의 lambda 처럼 익명함수가 있는데, 이를 closure라고 한다.
    // 클로저는 파라미터를 || 사이에 선언하고, 그 뒤에 함수에서 리턴하는 부분을 작성한다.
    let my_func = | x | x + 1;
    println!("{}", my_func(3));

    // 익명함수에서도 입, 출력 자료형을 명시해 줄 수 있다.
    // 자료형을 명시할 경우에 중괄호로 로직부분을 감싼다.
    let my_func = | x: i32| -> i32 { x + 1 };
    println!("{}", my_func(999));

    // 파이썬의 익명함수와는 달리 rust의 익명함수는 여러줄에 걸쳐 작성할 수 있다.
    let my_func = |mut x: i32| {
        x = x + 1;
        println!("{}", x);   
    };
    my_func(9999);
}