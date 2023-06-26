program loops
    implicit none
    integer :: i, sum

    ! 1부터 10까지의 합 계산
    sum = 0
    do i = 1, 10
        sum = sum + i
    end do
    write(*,*) "Sum:", sum

    ! 배열 순회
    integer, dimension(5) :: nums
    nums = [1, 2, 3, 4, 5]
    write(*,*) "Array elements:"
    do i = 1, 5
        write(*,*) nums(i)
    end do
end program loops

