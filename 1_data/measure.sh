for x in `seq 1 1000`
do
	#start=`date +%N`
	
	python 1-1_data.py
	#python 1-2_data_mpi.py
	
	#{ time ./a.out ; } 2>> log

	#finish=`date +%N`
	#diff=$( echo "$finish - $start" | bc -l )
	#echo "$x\t$diff"
done
