#1. 참고 자료
## rust 설치법
 ``` bash
    $curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    $source $HOME/.cargo/env
    $rustc --version
 ```

- [[자료 출처]](https://www.rust-lang.org/tools/install)

## jupyter에 rust kernel 추가하는 법
 ``` bash
    $cargo install evcxr_repl
    $conda install -y -c conda-forge nb_conda_kernels
    $cargo install evcxr_jupyter
    $evcxr_jupyter --install
 ```

 - [[자료 출처]](https://depth-first.com/articles/2020/09/21/interactive-rust-in-a-repl-and-jupyter-notebook-with-evcxr/)

 ## rust 공부 자료
 - 파이썬 프로그래머를 위한 러스트 입문 | [[링크]](https://indosaram.github.io/rust-python-book/ch2-02.html)

