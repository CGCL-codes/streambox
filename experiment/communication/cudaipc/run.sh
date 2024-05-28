rm -f testfifo*
rm -f app1.txt
rm -f app2.txt
rm app1
rm app2
nvcc -lcuda -o app1 app1.cu
nvcc -lcuda -o app2 app2.cu
nvprof ./app1 > app1.txt &
pid1=$!
nvprof ./app2 > app2.txt &
pid2=$!
wait $pid1
wait $pid2