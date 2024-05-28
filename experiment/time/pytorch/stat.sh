#!/bin/bash

# Function to calculate and print median, average, min, and max
# from a file of space-separated values.
function calc_stats {
    # Check if file exists
    if [ ! -f $1 ]; then
        echo "File not found: $1"
        return 1
    fi

    # The script removes "ms" and space characters, sorts the values,
    # then calculates the median, average, min, and max,
    # excluding values more than 1.5 IQR from the median (outliers).
    cat $1 | sed 's/ ms//g' | tr -d ' ' | sort -g | awk '
    {
        a[NR]=$1; 
    } 
    END {
        n = NR;
        if (n % 2) {
            median=a[int((n+1)/2)];
        } else {
            median=(a[n/2] + a[n/2+1])/2;
        }
        q1=a[int(n/4+1)];
        q3=a[int(3*n/4)+1];
        iqr=q3-q1;
        lower=q1-1.5*iqr;
        upper=q3+1.5*iqr;
        for(i=1; i<=n; i++) {
            if (a[i] < lower || a[i] > upper) {
                delete a[i];
            }
        }
        sum=0;
        cnt=0;
        for(i=1; i<=n; i++) {
            if(i in a){
                if(cnt == 0 || a[i] < min) min = a[i];
                if(cnt == 0 || a[i] > max) max = a[i];
                sum+=a[i];
                cnt++;
            }
        }
        avg=sum/cnt;
        print "Median: " median;
        print "Average: " avg;
        print "Min: " min;
        print "Max: " max;
    }'
}

# Calculate and print stats for each file
for file in context_init_times_torch.txt tensor_alloc_times_torch.txt model_load_times_torch.txt; do
    echo "Stats for $file:"
    calc_stats $file || echo "Error calculating stats for $file"
    echo
done
