program sin_function
  implicit none
  real :: x, result
  integer :: i, n
  real, parameter :: pi = 3.1415926535897932384626433832795
  
  ! 설정값
  n = 1000  ! 구간을 몇 등분할 것인지 결정
  
  ! 계산
  do i = 0, n
    x = 2.0 * pi * i / n
    result = sin(x)
   !write(*, '(F8.4, F10.6)') x, result
  end do
  
end program sin_function

