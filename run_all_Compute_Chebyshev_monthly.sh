# conda activate gUstNET

# loop over years 2011 to 2020
for year in {2020..2012}; do
    for month in {1..12}; do
        # combine all CERRA height level files for each month
        echo "Computing Chebyshev coefficients for $year-$month"

        python Compute_Chebyshev_coeffiients_monthly_to_zarr.py $year $month

    done
done

echo "All CERRA height level files combined to zarr"
