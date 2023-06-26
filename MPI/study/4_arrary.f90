program array_operations
    implicit none
    integer, dimension(5) :: nums
    real, dimension(3, 3) :: matrix

    ! 1차원 배열 초기화
    nums = [1, 2, 3, 4, 5]

    ! 2차원 배열 초기화
    matrix = reshape([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])

    ! 1차원 배열 출력
    write(*,*) "1D Array: ", nums

    ! 2차원 배열 출력
    write(*,*) "2D Matrix:"
    do i = 1, 3
        write(*,*) matrix(i, :)
    end do

    ! 배열 요소 접근과 수정
    write(*,*) "nums(3) =", nums(3)
    matrix(2, 2) = 10.0
    write(*,*) "matrix(2, 2) =", matrix(2, 2)
end program array_operations

