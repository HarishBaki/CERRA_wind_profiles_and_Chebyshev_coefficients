# conda activate gUstNET

# loop over years 2011 to 2020
for year in {2020..2011}; do
    for month in {1..12}; do
        # combine all CERRA height level files for each month
        echo "Combining CERRA height level files for $year-$month"

        # Since the monthly files are in HDD, they are taking too much time to read.
        # Thus, copy them first to SSD and then combine them.
        # Use rclone copy and from-files

        # Create a files.txt for the given year and month
        echo "Generating files.txt for $year-$month..."
        
        cat <<EOL > files.txt
CERRA_ws10/${year}/CERRA_${year}_${month}.nc
CERRA_ws10_step1/${year}/CERRA_${year}_${month}.nc
CERRA_ws10_step2/${year}/CERRA_${year}_${month}.nc
CERRA_ws100/${year}/CERRA_gridded_100_m_wind_${year}_${month}.nc
CERRA_ws150/${year}/CERRA_gridded_150_m_wind_${year}_${month}.nc
CERRA_ws_100_150_step1/${year}/CERRA_gridded_wind_${year}_${month}_1.nc
CERRA_ws_100_150_step2/${year}/CERRA_gridded_wind_${year}_${month}_2.nc
CERRA_ws_15_30_50_75_200_250_300_400_500/${year}/CERRA_gridded_15_30_50_75_200_250_300_400_500_wind_${year}_${month}.nc
CERRA_ws_15_30_50_75_200_250_300_400_500_step1/${year}/CERRA_gridded_15_30_50_75_200_250_300_400_500_wind_${year}_${month}_1.nc
CERRA_ws_15_30_50_75_200_250_300_400_500_step2/${year}/CERRA_gridded_15_30_50_75_200_250_300_400_500_wind_${year}_${month}_2.nc
EOL

        echo "Copying files for $year-$month to SSD..."
        rclone copy --progress --transfers 12 --files-from files.txt /media/harish/External_3/ /data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/

        python Combine_CERRA_height_level_monthly_chunks_to_zarr.py $year $month

        # remove the /data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp folder
        rm -r /data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp
    done
done

echo "All CERRA height level files combined to zarr"
