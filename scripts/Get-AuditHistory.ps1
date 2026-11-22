<#
Download the fixed public research fixture used in AUDIT.md.
Price fields are scaled by Yahoo Finance's adjusted close / close ratio.
This script uses public HTTPS market data only. It has no broker credentials.
#>
param([string]$Output = 'audit-history.csv')
$ErrorActionPreference = 'Stop'
$rows = foreach ($symbol in @('SPY', 'AAPL', 'MSFT', 'JPM', 'XOM')) {
    $uri = "https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=1420070400&period2=1767225600&interval=1d"
    $response = Invoke-RestMethod -Uri $uri -UserAgent 'Mozilla/5.0' -TimeoutSec 30
    $chart = $response.chart.result[0]
    if ($chart.meta.symbol -ne $symbol -or $chart.timestamp.Count -lt 1000) {
        throw "Incomplete daily history for $symbol"
    }
    $quote = $chart.indicators.quote[0]
    $adjusted = $chart.indicators.adjclose[0].adjclose
    for ($index = 0; $index -lt $chart.timestamp.Count; $index++) {
        $close = $quote.close[$index]
        if ($null -eq $quote.open[$index] -or $null -eq $quote.high[$index] -or
            $null -eq $quote.low[$index] -or $null -eq $quote.volume[$index] -or
            $close -le 0 -or $adjusted[$index] -le 0) { continue }
        $ratio = [double]$adjusted[$index] / $close
        [pscustomobject]@{
            symbol = $symbol
            date = [DateTimeOffset]::FromUnixTimeSeconds($chart.timestamp[$index]).UtcDateTime.ToString('yyyy-MM-dd')
            open = $quote.open[$index] * $ratio
            high = $quote.high[$index] * $ratio
            low = $quote.low[$index] * $ratio
            close = [double]$adjusted[$index]
            volume = $quote.volume[$index]
        }
    }
}
$rows | Export-Csv -LiteralPath $Output -NoTypeInformation -Encoding ascii
