program data_generator
  use mpi
  implicit none
  integer :: ierr, rank, size, i, n, local_n, local_i, numSteps, start_, end_, step_ , localStart, localEnd
  real(8) :: x,   start_time, end_time, elapsed_ , k
  real(8), dimension(:), allocatable :: x_local, result_local, result
  real, parameter :: pi = 3.1415926535897932384626433832795
  real*8, external :: f1
  
  call cpu_time(start_time)
  ! 초기화
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
  
 
  start_ = 0.0
  end_ = 10000000! 10000.0
  step_ = 1 ! 0.001




  ! 설정값
  numSteps = INT(((end_ - start_) / (step_ * size)))  ! 구간을 몇 등분할 것인지 결정
  local_n = n / size
  
  allocate(x_local(numSteps), result_local(numSteps), result(numSteps))
  
  ! 각 프로세스에게 할당된 범위 계산
  localStart = start_ + rank * numSteps * step_
  !print *, localStart
  localEnd = localStart + numSteps * step_
  !print *, localEnd
  ! 계산
  do i = 1, numSteps, step_
    k = (i+localStart)*0.001
    result_local(i) = f1(k)
    !print *, k, result_local(i)
  end do
  
  ! 결과 수집
  call MPI_Gather(result_local, numSteps, MPI_REAL, result, numSteps, MPI_REAL, 0, MPI_COMM_WORLD, ierr)
  
  ! 결과 출력
  !if (rank == 0) then
  !  do i = 1, n
  !    x = 
  !    write(*, '(F8.4, F10.6)') x, result(i)
  !  end do
  !end if
  
  deallocate(x_local, result_local)
  
  ! MPI 종료
  call MPI_Finalize(ierr)
  call cpu_time(end_time)

  elapsed_ = end_time - start_time
  if (rank == 0) then
      print *, elapsed_
  end if
  
end program data_generator

real(8) function f1(a)
  real(8),intent(in) :: a
  f1 = a**2 - 3 * a + 2
end function f1
