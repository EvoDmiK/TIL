fn main(){
    let x = 1.0;
    let y =  10;

    // rust에서는 두 값을 비교하기 위해선 두 자료형이 같아야해서
    // 타입 캐스팅을 해줘야 한다.
    if x < (y as f64) {
        println!("x is less than y");
    }
    else if x == (y as f64){
        println!("x is equal to y");
    }
    else {
        println!("x is greater than y");
    }
}