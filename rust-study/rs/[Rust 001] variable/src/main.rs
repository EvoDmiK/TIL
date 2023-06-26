fn main() {
    //  파이썬과 달리 rust는 매크로(macro)를 이용해 값을 출력하고,
    //  함수 이름 뒤에 항상 !가 붙으며, 코드의 마지막에는 ;이 붙는다.
    //! 굉장히 킹받는 점은 출력문 안에 작은 따옴표 사용이 불가능하다.
    println!("Hello, World");

    // rust에서 변수를 선언하기 위해서는 아래와 같은 규칙을 가진다.
    // let 변수명: 자료형 = 값; (단, 자료형은 생략이 가능하다.)
    let x: i32 = 10;
    println!("x        = {}", x);

    let y = 20;
    println!("y        = {}", y);

    //rust 에서는 변수의 불변성이라는게 있어서 let으로 선언된 변수는 
    // 다른 값이 적용이 안되어 let 뒤에 mut을 붙여줘야한다.
    let mut x = 20;
    println!("before x = {}", x);

    x = 30;
    println!("aftre x  = {}", x);

    // 변수의 값을 변경 할 수는 없지만, 변수를 새로 선언하는 것이 가능하다.
    // 이름을 재사용해 새로운 변수를 선언하는 것을 셰도잉이라 한다.
    let x_ = 5;
    println!("before x = {}", x_);

    let x_ = "6";
    println!("after x  = {}", x_);

    // 타입 캐스팅
    // rust에서는 as 를 통해 타입을 변경할 수 있다.
    let x: f64 = 1.2;
    let y = x as i32;
    println!("x : {} -> y : {}", x, y);
}
