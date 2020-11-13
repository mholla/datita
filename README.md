# datita
visualization of baby data from Hatch Baby app

This code is intended to provide beautiful and informative visualization of your baby's patterns and changes in feeding, sleep, diapers, and growth.  I created it because I found the default visualizations to be unhelpful.  Specifically, I had a few questions: 
* SLEEP 
  * how long is my baby sleeping each day? (And here I find it important to calculate this per 24 hours, not considering any sleep that _starts_ before midnight to be part of that day) -> sleep_total.png
  * is my baby following a sleep schedule?  How consistent is it?  Does it differ between day and nighttime?  -> sleep_schedule.png
  * What are their typical wake windows?  Do they differ between day and nighttime? -> sleep_wake.png
  * What is the best bedtime for my baby?  What is their maximum uninterrupted sleep time, and when does it start? -> sleep_duration.png and sleep_max.png
* FEEDING
  * How much are they eating every day?  -> feeding_totals.png
  * How much are they eating relative to their weight? -> feeding_percentage.png
  * What is their pace of feeding throughout the day? (Useful to have an idea if they are "behind schedule" at some point in the day) -> feeding_over_time.png
  * How much are they eating per feeding? (Sometimes I log several feeding events from the same bottle, or with a refill.  What I want to know is, how much should the bottle have started with?) -> feeding_amount_per.png
  * Is the amount the eat related to how long it's been since they ate last?  (Not strongly enough to be helpful) -> feeding_amount_interval.png
  * How much are they eating overnight?  (Potentially helpful to inform night weaning) -> feeding_time_of_day.png
* DIAPERS (note: I only analyze dirty diapers because wet diapers are much more a function of when we change them than when they occur)
  * how many dirty diapers do they have each day? (This was an important reassurance for us when we stopped using a diaper service and switched to using our own cloth diapers) -> diapers_number.png
  * when do dirty diapers tend to occur?  (This is helpful in knowing how important middle-of-the-night changes will be, and in case you're interested in elimination communication) -> diapers_time.png
* GROWTH
  * how are my baby's weight, length, and head circumference changing over time and relative to WHO percentiles? -> weight.png, length.png, and head.png



## Technical details ##

This code accepts a text file exported from the Hatch Baby app and produces a number of visualizations.  It could be modified to accept exports from other apps fairly easily, I believe, by editing the parsing functions.

The only variables that are hard-coded are: 
* the filename of the Hatch export
* a separate file containing head circumference data if you want to analyze that as well
* the gender (for percentiles)
* the birthdate
* your desired colormap
* morning and night times (i.e., at what time does the "day" start and end for you?)

Percentiles are taken from the CDC website, following their suggestion to use WHO data from 0-24 months (https://www.cdc.gov/growthcharts/who_charts.htm)
