program conditionals
    implicit none
    integer :: num

    write(*,*) "Enter a number:"
    read(*,*) num

    if (num > 0) then
        write(*,*) "The number is positive."
    else if (num < 0) then
        write(*,*) "The number is negative."
    else
        write(*,*) "The number is zero."
    end if

    select case (num)
        case (1, 2, 3)
            write(*,*) "The number is 1, 2, or 3."
        case (4, 5, 6)
            write(*,*) "The number is 4, 5, or 6."
        case default
            write(*,*) "The number is not 1-6."
    end select
end program conditionals

