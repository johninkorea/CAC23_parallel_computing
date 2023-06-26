program variables
    implicit none
    integer :: num1, num2
    real :: floatNum
    character(10) :: name

    num1 = 10
    num2 = 20
    floatNum = 3.14
    name = "John Doe"

    write(*,*) "num1 =", num1
    write(*,*) "num2 =", num2
    write(*,*) "floatNum =", floatNum
    write(*,*) "name =", name
end program variables

