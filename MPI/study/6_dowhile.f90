program do_while_loop
    implicit none
    integer :: num, sum

    ! 사용자로부터 양의 정수를 입력받아 합 구하기
    sum = 0
    num = 0
    do while (num >= 0)
        write(*,*) "Enter a positive integer (negative to exit):"
        read(*,*) num
        if (num >= 0) then
            sum = sum + num
        end if
    end do
    write(*,*) "Sum:", sum
end program do_while_loop

