program sin_function_mpi
  use mpi
  implicit none
  integer :: ierr, rank, size, i, n, local_n, local_i
  real :: x, result
  real, dimension(:), allocatable :: x_local, result_local
  real, parameter :: pi = 3.1415926535897932384626433832795
  
  ! 초기화
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
  
  ! 설정값
  n = 1000  ! 구간을 몇 등분할 것인지 결정
  local_n = n / size
  
  allocate(x_local(local_n), result_local(local_n))
  
  ! 각 프로세스에게 할당된 범위 계산
  do i = 1, local_n
    local_i = (rank * local_n) + i - 1
    x_local(i) = 2.0 * pi * local_i / n
  end do
  
  ! 계산
  do i = 1, local_n
    result_local(i) = sin(x_local(i))
  end do
  
  ! 결과 수집
  call MPI_Gather(result_local, local_n, MPI_REAL, result, local_n, MPI_REAL, 0, MPI_COMM_WORLD, ierr)
  
  ! 결과 출력
  if (rank == 0) then
    do i = 1, n
      x = 2.0 * pi * (i - 1) / n
      write(*, '(F8.4, F10.6)') x, result(i)
    end do
  end if
  
  deallocate(x_local, result_local)
  
  ! MPI 종료
  call MPI_Finalize(ierr)
  
end program sin_function_mpi

