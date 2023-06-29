program data_generator
  implicit none
  real(8) :: x, result,  start_, end_, step_, start_time, end_time, elapsed_
  integer :: i, n
  real, parameter :: pi = 3.1415926535897932384626433832795
  real*8, external :: f1
  
  call cpu_time(start_time)
  start_ = 0.0
  end_ = 100.0
  step_ = 0.001
  ! 설정값
  !n = INT((end_ - start_)/step_)  ! 구간을 몇 등분할 것인지 결정

  result = 0.0
  ! 계산
  do x = start_, end_, step_ 
    result = f1(x)
   !write(*, '(F8.4, F10.6)') x, result
  end do
  
  call cpu_time(end_time)

  elapsed_ = end_time - start_time

  print *, elapsed_
end program data_generator
real(8) function f1(a)
  real(8),intent(in) :: a
  f1 = a**2 - 3 * a + 2
end function f1