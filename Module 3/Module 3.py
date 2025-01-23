import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the parquet file from the URL into a Pandas DataFrame
df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet')

# Print the first 5 rows of data. Study the schema and make sure you understand what each of the fields mean by referencing the documentation
print(df.head())
dfcolumns = list(df.columns)
print(dfcolumns)

# How many rows are in the dataset? How many unique columns are in the dataset?
print(f"Number of rows: {df.shape[0]}")
print(f"Number of unique columns: {df.shape[1]}")
# Rows: 2,463,931
# Columns: 19

# Which columns have NULL values and how many NULL values are present in each of these columns?
print("Columns with NULL values and their counts:")
print(df.isnull().sum())

# Drop rows with NULL values
df_no_nulls = df.dropna()

pd.set_option('display.max_columns', None)
pd.set_option('float_format', '{:f}'.format)
print ("Summary Statistics as follows:\n")
#Generate summary statistics using Pandas' describe method. Do you notice anything unusual in the dataset?
#Find at least one anomaly and try to come up with a hypothesis to explain it.

#Anamoly-1: Minimum Passenger count = 0.  How can this be possible to get a taxi ride without any passenger?
#Possible-Explanation: Meter might have been running when there was no passenger
#which could have resulted in a data being recorded as a trip erroneously.

#Anamoly-2: Negative value in Min for fare_amount, extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge.
#How is possible to get a taxi-ride where the cab-driver is not getting paid and the passenger is receiving the money?
#Possible-Explanation: Perhaps passenger is getting refunded due to disputes?

#print("Summary statistics:")
#print(df['VendorID'].describe())
#print(df['tpep_pickup_datetime'].describe())
#print(df['tpep_dropoff_datetime'].describe())
#print(df['passenger_count'].describe())
#print(df['trip_distance'].describe())
print(df.describe())
#print(df.describe(include='all'))
#Create a new feature that calculates the trip duration in minutes.
df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
#Create additional features for the pick-up day of week and pick-up hour.
df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

#Use the Seaborn library to create a line plot depicting the number of trips as a function of the hour of day. What's the busiest time of day?
#Busiest time of day is between 18:00 - 18:20 in the evening.
#Create another lineplot depicting the number of trips as a function of the day of week. What day of the week is the least busy?
#Tuesday(1) is the least busy day of the week in terms on number of trips.

#Group data by pickup hour and count the number of trips
hourly_trips = df.groupby('pickup_hour')['pickup_hour'].count()
# Create the line plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x=hourly_trips.index, y=hourly_trips.values)
# Find the maximum y-value and its corresponding x-value
max_y = hourly_trips.values.max()
max_x = hourly_trips.index[hourly_trips.values == max_y].values[0]
print("Maximum number of Trips", max_y, "Hour with max trips", max_x)
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.title('Number of Trips by Hour of Day')
#Add a marker for the maximum y-value
plt.plot(max_x, max_y, 'ro')
#Optionally, add a text label
plt.text(max_x, max_y, f'({max_x}, {max_y})', ha='center', va='bottom')
plt.show()

#Create another lineplot depicting the number of trips as a function of the day of week. What day of the week is the least busy?
daily_trips = df.groupby('pickup_day_of_week')['pickup_day_of_week'].count()
# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=daily_trips.index, y=daily_trips.values)
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Number of Trips')
plt.title('Number of Taxi Trips by Day of Week')
plt.show()

# Compute correlation matrix
correlation_matrix = df[['trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 'trip_duration_minutes']].corr()

# Create heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Trip Variables')
plt.show()

# Sample the dataset for pairplot (if memory issues)
df_sample = df[['trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 'trip_duration_minutes']].sample(n=10, random_state=42)

# Create a pairplot using Seaborn
sns.pairplot(df_sample)
plt.show()

#Use Seaborn to create a countplot for the variables PULocationID, and DOLocationID. Keep only the top 15 pick-up and drop-off locations. What's the most popular pick-up location?
# Count plot for PULocationID (top 15)
top_15_pickup = df['PULocationID'].value_counts().nlargest(15).index
plt.figure(figsize=(12, 6))
sns.countplot(x='PULocationID', data=df[df['PULocationID'].isin(top_15_pickup)])
plt.title('Top 15 Pickup Locations')
plt.xlabel('PULocationID')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.show()

# Find the most popular pick-up location
most_popular_pickup = df['PULocationID'].value_counts().idxmax()
print(f"The most popular pick-up location is: {most_popular_pickup}")

# Count plot for DOLocationID (top 15)
top_15_dropoff = df['DOLocationID'].value_counts().nlargest(15).index
plt.figure(figsize=(12, 6))
sns.countplot(x='DOLocationID', data=df[df['DOLocationID'].isin(top_15_dropoff)])
plt.title('Top 15 Drop-off Locations')
plt.xlabel('DOLocationID')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.show()

#Use Seaborn's boxplot to discern the relationship between payment_type and total_amount. Does anything look weird? Can you explain what's going on?
#There is an outlier for a very large amount (400,000) for Payment Type 4 (Dispute).  Perhaps a law-suit or a large settlement from law-suit.
#When the outlier is removed, the plot is more clear showing the total_amount distribution. Some Disputes are resulting in negative total amount,
#no voids are found.  There is some total amount against Payment Type = 0 which does not have a definition in data dictionary.
#Create a box plot of total amount by payment type
df_sample_1 = df.sample(n=10, random_state=42)  # Adjust n as needed
#plt.figure(figsize=(20, 12))
#sns.boxplot(x='payment_type', y='total_amount', data=df_sample_1)
sns.boxplot(x='payment_type', y='total_amount', data=df)
#sns.boxplot(x='total_amount',y='payment_type',data=df)
plt.title('Total Amount Distribution by Payment Type')
plt.xlabel('Payment Type')
plt.ylabel('Total Amount')
plt.show()

sns.boxplot(x='payment_type', y='total_amount', data=df, showfliers=False)
plt.title('Total Amount Distribution by Payment Type')
plt.xlabel('Payment Type')
plt.ylabel('Total Amount')
plt.show()

#Use Seaborn's histplot to explore the data distributions for fare_amount, trip_distance, and extra. Use kernel density estimators to better visualize the distribution. Use sampling if you run into any memory issues.
df_sample_2 = df.sample(n=100, random_state=42)  # Adjust n as needed

# Create histograms with KDE for fare_amount, trip_distance, and extra
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df_sample_2['fare_amount'], kde=True)
plt.title('Fare Amount Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df_sample_2['trip_distance'], kde=True)
plt.title('Trip Distance Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df_sample_2['extra'], kde=True)
plt.title('Extra Charge Distribution')

plt.tight_layout()
plt.show()