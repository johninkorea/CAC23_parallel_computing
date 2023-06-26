program operators
    implicit none
    integer :: num1, num2, sum

    num1 = 10
    num2 = 20

    sum = num1 + num2

    if (sum > 30) then
        write(*,*) "Sum is greater than 30."
    else if (sum == 30) then
        write(*,*) "Sum is equal to 30."
    else
        write(*,*) "Sum is less than 30."
    end if
end program operators

