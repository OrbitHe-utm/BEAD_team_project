# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, count, mean, stddev, to_date
# import matplotlib.pyplot as plt
# import seaborn as sns

# 初始化 SparkSession
def create_spark_session():
    spark = SparkSession.builder \
        .appName("H&M Data Analysis") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .getOrCreate()
    return spark

# 数据加载
def load_data(spark, raw_data_path):

    articles = spark.read.csv(f"{raw_data_path}/articles.csv", header=True, inferSchema=True)
    customers = spark.read.csv(f"{raw_data_path}/customers.csv", header=True, inferSchema=True)
    transactions = spark.read.csv(f"{raw_data_path}/transactions_train.csv", header=True, inferSchema=True)
    return articles, customers, transactions
def save_visualizations(spark, dir_path, articles, customers, transactions):
    # 查看 index_name 分布
    index_name_dist = articles.groupBy("index_name").count().orderBy(col("count").desc())
    print("Index Name Distribution:")
    index_name_dist.show()
    # 转换为 Pandas 并可视化 index_name 分布
    index_name_pd = index_name_dist.toPandas()
    plt.figure(figsize=(15, 7))
    sns.barplot(data=index_name_pd, y="index_name", x="count", color="orange")
    plt.xlabel("Count by Index Name")
    plt.ylabel("Index Name")
    plt.title("Distribution of Index Name")
    plt.show()

    # 查看 garment_group_name 分布
    garment_group_dist = articles.groupBy("garment_group_name", "index_group_name").count().orderBy(col("count").desc())
    print("Garment Group Distribution:")
    garment_group_dist.show()

    # 转换为 Pandas 并可视化 garment_group_name 分布
    garment_group_pd = garment_group_dist.toPandas()
    plt.figure(figsize=(15, 7))
    sns.barplot(
        data=garment_group_pd,
        x="count",
        y="garment_group_name",
        hue="index_group_name",
        dodge=False
    )
    plt.xlabel("Count by Garment Group")
    plt.ylabel("Garment Group")
    plt.title("Garment Group Distribution")
    plt.show()

    # 查看 index_group_name 和 index_name 的分组计数
    index_group_count = articles.groupBy("index_group_name", "index_name").count()
    print("Index Group and Index Name Count:")
    index_group_count.show()

    # 检查 customers 数据
    total_customers = customers.count()
    unique_customers = customers.select("customer_id").distinct().count()
    duplicates = total_customers - unique_customers
    print(f"Total Customers: {total_customers}, Unique Customers: {unique_customers}, Duplicates: {duplicates}")

    # 检查 postal_code 的异常记录
    postal_dist = customers.groupBy("postal_code").count().orderBy(col("count").desc())
    print("Postal Code Distribution:")
    postal_dist.show(5)

    # 查看 transactions 数据
    print("Transactions Price Statistics:")
    transactions.describe(["price"]).show()

    # 转换为 Pandas 并绘制价格箱线图
    transactions_pd = transactions.select("price").toPandas()
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=transactions_pd, x="price", color="orange")
    plt.xlabel("Price Outliers")
    plt.title("Boxplot of Prices")
    plt.show()

    # 合并 articles 和 transactions 数据
    articles_for_merge = articles.select("article_id", "prod_name", "product_type_name", "product_group_name", "index_name")
    transactions_for_merge = transactions.select("customer_id", "article_id", "price", "t_dat")
    articles_transactions = transactions_for_merge.join(articles_for_merge, on="article_id", how="left")
    print("Merged Data Preview:")
    articles_transactions.show(5)

    # 按 product_group_name 分析价格分布
    price_dist_pd = articles_transactions.select("price", "product_group_name").toPandas()
    plt.figure(figsize=(25, 18))
    sns.boxplot(data=price_dist_pd, x="price", y="product_group_name")
    plt.xlabel("Price Outliers")
    plt.ylabel("Product Group Name")
    plt.title("Price Distribution by Product Group")
    plt.show()

    # 查看大类商品的均价
    average_price_by_index = articles_transactions.groupBy("index_name").agg(mean("price").alias("avg_price"))
    average_price_pd = average_price_by_index.orderBy(col("avg_price").desc()).toPandas()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=average_price_pd, x="avg_price", y="index_name", color="orange", alpha=0.8)
    plt.xlabel("Average Price")
    plt.ylabel("Index Name")
    plt.title("Average Price by Index Name")
    plt.show()

    # 查看价格变化趋势
    articles_transactions = articles_transactions.withColumn("t_dat", to_date(col("t_dat"), "yyyy-MM-dd"))
    product_list = ['Shoes', 'Garment Full body', 'Bags', 'Garment Lower body', 'Underwear/nightwear']
    colors = ['cadetblue', 'orange', 'mediumspringgreen', 'tomato', 'lightseagreen']

    k = 0
    f, ax = plt.subplots(3, 2, figsize=(20, 15))
    for i in range(3):
        for j in range(2):
            try:
                product = product_list[k]
                product_data = articles_transactions.filter(col("product_group_name") == product)
                series_mean = product_data.groupBy("t_dat").agg(mean("price").alias("mean_price")).orderBy("t_dat").toPandas()
                series_std = product_data.groupBy("t_dat").agg(stddev("price").alias("std_price")).orderBy("t_dat").toPandas()

                ax[i, j].plot(series_mean["t_dat"], series_mean["mean_price"], linewidth=4, color=colors[k])
                ax[i, j].fill_between(
                    series_mean["t_dat"],
                    series_mean["mean_price"] - 2 * series_std["std_price"],
                    series_mean["mean_price"] + 2 * series_std["std_price"],
                    color=colors[k], alpha=0.1
                )
                ax[i, j].set_title(f"Mean {product} Price Over Time")
                ax[i, j].set_xlabel("Date")
                ax[i, j].set_ylabel("Price")
                k += 1
            except IndexError:
                ax[i, j].set_visible(False)

    plt.tight_layout()
    plt.show()



def main():
    # 查看数据结构
    print("Articles Schema:")
    articles.printSchema()
    print("Customers Schema:")
    customers.printSchema()
    print("Transactions Schema:")
    transactions.printSchema()
    # 查看 articles 数据
    print("First 5 rows of articles:")
    articles.show(5, truncate=False)
spark.stop()