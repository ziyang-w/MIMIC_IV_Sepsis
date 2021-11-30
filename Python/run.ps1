
$fileName = "final_sepsis.csv", "tt.csv",
$targetY = "death_within_30_days", "death_within_30_days"

for($i=0; $i -lt $fileName.Length; $i++)
{
    echo '========start running========' $fileName[$i]
    echo "targetY:" $targetY[$i]
    python processing.py $fileName[$i] $targetY[$i]
}

