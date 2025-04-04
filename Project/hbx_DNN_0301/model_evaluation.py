from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext


def clear_spark_cache():
    try:
        # 获取当前的 SparkContext
        sc = SparkContext.getOrCreate()
        # 停止当前的 SparkContext
        sc.stop()
        # 重新创建一个新的 SparkContext，这会清除之前的缓存
        SparkContext.setSystemProperty('spark.driver.memory', '2g')
        new_sc = SparkContext(appName='base model')
        # 为了后续使用 SparkSession，需要关闭新创建的 SparkContext
        new_sc.stop()
    except Exception as e:
        print(f"清除缓存时出现错误: {e}")


def load_data(spark, path):
    # read csv data
    transaction_raw = spark.read.parquet(path, inferSchema=True, header=True)
    # 对数据进行 1% 的采样
    sampled_data = transaction_raw.sample(fraction=0.05, withReplacement=False, seed=42)
    return sampled_data


def train_test_split(spark, df):
    df.createOrReplaceTempView('transactions')
    transactions_train = spark.sql("""
        with ranked as (
        select *,
        ntile(100) over (order by t_dat asc) as percentile
        from transactions
        )
        select * from ranked where percentile <= 80
        """)
    transactions_test = spark.sql("""
    with ranked as (
        select *,
        ntile(100) over (order by t_dat asc) as percentile
        from transactions
        )
        select * from ranked where percentile > 80
    """)
    return {'train': transactions_train,
            'test': transactions_test}


def get_pop_10(spark, df):
    df.createOrReplaceTempView('train')
    pop_10 = spark.sql("""
    with summary as (
    select article_id, count(article_id) as cnt
    from train
    group by article_id
    )
    select article_id from summary order by cnt desc limit 10
    """)
    return [row.article_id for row in pop_10.collect()]


def rec_pop_ten(spark, df, pop_list):
    return df.withColumn('base_rec', F.array([F.lit(x) for x in pop_list]))


def model_eval(spark, outcome_df, rec_col, usr_col):
    # 计算基础模型在测试集上的击中率
    base_hit_count = outcome_df \
        .withColumn('is_hit', F.array_contains(F.col(rec_col), F.col('article_id'))) \
        .groupBy(usr_col) \
        .agg(F.sum(F.col('is_hit').cast('int')).alias('hit_count')) \
        .agg(F.avg(F.col('hit_count') / 10).alias('base_hit_rate')).collect()[0][0]

    # 输出比较矩阵
    result_df = spark.createDataFrame([
        ('Base Model', base_hit_count)
    ], ['Model', 'Hit Rate'])

    return result_df


def main():
    # 清除 Spark 缓存
    clear_spark_cache()
    spark = SparkSession.builder.appName('base model').master('local[*]').getOrCreate()
    hdfs_path = 'hdfs://localhost:9000/data/transactions_train'
    data = load_data(spark, hdfs_path)
    data.show(10)
    split = train_test_split(spark, data)
    train, test = split['train'], split['test']
    # 获取训练集的最热门前 10 商品 id
    pop_ten = get_pop_10(spark, train)
    # 为测试集添加基础模型推荐列
    test = rec_pop_ten(spark, test, pop_ten)

    result_df = model_eval(spark, test, 'base_rec', 'customer_id')
    result_df.show()
    spark.stop()


if __name__ == "__main__":
    main()
