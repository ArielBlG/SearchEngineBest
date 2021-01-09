
if __name__ == '__main__':
    import os
    import sys
    import re
    from datetime import datetime
    import pandas as pd
    import pyarrow.parquet as pq
    import time
    import timeit
    import importlib
    import logging
    import pandas as pd
    import metrics
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import pickle


    def recall(df, docs_retreived, q_id):
        """
            This function will calculate the recall of a specific query or of the entire DataFrame
            :param df: DataFrame: Contains query numbers, tweet ids, and label
            :param num_of_relevant: Dictionary: number of relevant tweets for each query number. keys are the query number and values are the number of relevant.
            :return: Double - The recall
        """
        correct_counter = 0
        for doc in docs_retreived:
            if doc in df[q_id][1]:
                correct_counter += 1
        return correct_counter / df[q_id][0]


    def test_file_exists(fn):
        if os.path.exists(fn):
            return True
        logging.error(f'{fn} does not exist.')
        return False


    tid_ptrn = re.compile('\d+')


    def invalid_tweet_id(tid):
        if not isinstance(tid, str):
            tid = str(tid)
        if tid_ptrn.fullmatch(tid) is None:
            return True
        return False


    bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')
    bench_lbls_path = os.path.join('data', 'benchmark_lbls_train.csv')
    queries_path = os.path.join('data', 'queries_train.tsv')
    model_dir = os.path.join('.', 'model')

    results = {} # (map,precision,precision@5,precision@10,precision@50,recall)


    bench_lbls = None
    q2n_relevant = None
    if not test_file_exists(bench_data_path) or \
            not test_file_exists(bench_lbls_path):
        #logging.error("Benchmark data does exist under the 'data' folder.")
        sys.exit(-1)
    else:
        bench_lbls = pd.read_csv(bench_lbls_path,
                                 dtype={'query': int, 'tweet': str, 'y_true': int})
        q2n_relevant = bench_lbls.groupby('query')['y_true'].sum().to_dict()
        #logging.info("Successfully loaded benchmark labels data.")

    # is queries file under data?
    queries = None
    if not test_file_exists(queries_path):
        pass
    else:
        queries = pd.read_csv(os.path.join('data', 'queries_train.tsv'), sep='\t')
        pass

    import configuration

    config = configuration.ConfigClass()

    # test for each search engine module
    all_results = {}
    engine_modules = ['search_engine_' + name for name in ['1','2','3','4','5','best']] # TODO: ADD ALL YOUR ENGINES HERE
    for engine_module in engine_modules:

        # does the module file exist?
        if not test_file_exists(engine_module + '.py'):
            continue
        # try importing the module
        se = importlib.import_module(engine_module)
        engine = se.SearchEngine(config=config)

        # test building an index and doing so in <1 minute
        build_idx_time = timeit.timeit(
            "engine.build_index_from_parquet(bench_data_path)",
            globals=globals(), number=1
        )

        engine.load_precomputed_model(model_dir)

        # run multiple queries and test that no query takes > 10 seconds
        with open(r'for_recall', 'rb') as f:
            rec = pickle.load(f)
        if queries is not None:
            for i, row in queries.iterrows():
                queries_results = []
                q_id = row['query_id']
                q_keywords = row['keywords']
                start_time = time.time()
                q_n_res, q_res = engine.search(q_keywords)
                end_time = time.time()
                q_time = end_time - start_time
                if q_n_res is None or q_res is None or q_n_res < 1 or len(q_res) < 1:
                    pass
                else:
                    invalid_tweet_ids = [doc_id for doc_id in q_res if invalid_tweet_id(doc_id)]
                    queries_results.extend(
                        [(q_id, str(doc_id)) for doc_id in q_res if not invalid_tweet_id(doc_id)])

                queries_results = pd.DataFrame(queries_results, columns=['query', 'tweet'])

                # merge query results with labels benchmark
                q_results_labeled = None
                if bench_lbls is not None and len(queries_results) > 0:
                    q_results_labeled = pd.merge(queries_results, bench_lbls,
                                                 on=['query', 'tweet'], how='inner', suffixes=('_result', '_bench'))
                    # q_results_labeled.rename(columns={'y_true': 'label'})
                    zero_recall_qs = [q_id for q_id, rel in q2n_relevant.items() \
                                      if metrics.recall_single(q_results_labeled, rel, q_id) == 0]

                if q_results_labeled is not None:
                    # test that MAP > 0
                    results_map = metrics.map(q_results_labeled)
                    # test that the average across queries of precision,
                    # precision@5, precision@10, precision@50, and recall
                    # is in [0,1].
                    prec, p5, p10, p50 = \
                        metrics.precision(q_results_labeled), \
                        metrics.precision(q_results_labeled.groupby('query').head(5)), \
                        metrics.precision(q_results_labeled.groupby('query').head(10)), \
                        metrics.precision(q_results_labeled.groupby('query').head(50))
                    recalls = recall(rec, q_res, q_id)

                    all_results[(engine_module,q_id)] = ((prec, p5, p10, p50, recalls),q_keywords)
    for q_id in range(1, 36):
        # Initialize a figure with ff.create_table(table_data)
        fig = go.Figure()

        # Add graph data
        teams = ['search_engine_1', 'search_engine_2', 'search_engine_3',
                 'search_engine_4', 'search_engine_5', 'search_engine_best']

        prec = [all_results[(teams[0], q_id)][0][0], all_results[(teams[1], q_id)][0][0], all_results[(teams[2], q_id)][0][0],
                all_results[(teams[3], q_id)][0][0], all_results[(teams[4], q_id)][0][0], all_results[(teams[5], q_id)][0][0]]
        p5 = [all_results[(teams[0], q_id)][0][1], all_results[(teams[1], q_id)][0][1], all_results[(teams[2], q_id)][0][1],
              all_results[(teams[3], q_id)][0][1], all_results[(teams[4], q_id)][0][1], all_results[(teams[5], q_id)][0][1]]
        p10 = [all_results[(teams[0], q_id)][0][2], all_results[(teams[1], q_id)][0][2], all_results[(teams[2], q_id)][0][2],
               all_results[(teams[3], q_id)][0][2], all_results[(teams[4], q_id)][0][2], all_results[(teams[5], q_id)][0][2]]
        p50 = [all_results[(teams[0], q_id)][0][3], all_results[(teams[1], q_id)][0][3], all_results[(teams[2], q_id)][0][3],
               all_results[(teams[3], q_id)][0][3], all_results[(teams[4], q_id)][0][3], all_results[(teams[5], q_id)][0][3]]
        recall = [all_results[(teams[0], q_id)][0][4], all_results[(teams[1], q_id)][0][4], all_results[(teams[2], q_id)][0][4],
                  all_results[(teams[3], q_id)][0][4], all_results[(teams[4], q_id)][0][4], all_results[(teams[5], q_id)][0][4]]

        dict_to_csv = {}
        print(f'Query id:{q_id}')
        print("precision")
        print(prec)
        dict_to_csv["prec"]= prec
        print("p5")
        print(p5)
        dict_to_csv["p5"]= p5
        print("p10")
        print(p10)
        dict_to_csv["p10"]= p10
        print("p50")
        print(p50)
        dict_to_csv["p50"]= p50
        print("recall")
        print(recall)
        dict_to_csv["recall"]= recall

        # df_to_csv = pd.DataFrame(dict_to_csv)
        # if not os.path.exists("metrics_output"):
        #     os.mkdir("metrics_output")
        #
        # df_to_csv.to_csv(f"./metrics_output/{q_id}.csv")

        # Make traces for graph TODO: Change color pallete with the site https://www.coolors.co/generate
        trace1 = go.Bar(x=teams, y=prec, xaxis='x2', yaxis='y2',
                        marker=dict(color='#717C89'),
                        name='Precision')
        trace2 = go.Bar(x=teams, y=p5, xaxis='x2', yaxis='y2',
                        marker=dict(color='#8AA2A9'),
                        name='Precision@5')
        trace3 = go.Bar(x=teams, y=p10, xaxis='x2', yaxis='y2',
                        marker=dict(color='#90BAAD'),
                        name='Precision@10')
        trace4 = go.Bar(x=teams, y=p50, xaxis='x2', yaxis='y2',
                        marker=dict(color='#A1E5AB'),
                        name='Precision@50')
        trace5 = go.Bar(x=teams, y=recall, xaxis='x2', yaxis='y2',
                        marker=dict(color='#ADF6B1'),
                        name='Recall')

        # Add trace data to figure
        fig.add_traces([trace1, trace2, trace3, trace4, trace5])

        # initialize xaxis2 and yaxis2
        fig['layout']['xaxis2'] = {}
        fig['layout']['yaxis2'] = {}

        # Edit layout for subplots
        fig.layout.yaxis.update({'domain': [0, .45]})
        fig.layout.yaxis2.update({'domain': [.6, 1]})

        # The graph's yaxis2 MUST BE anchored to the graph's xaxis2 and vice versa
        fig.layout.yaxis2.update({'anchor': 'x2'})
        fig.layout.xaxis2.update({'anchor': 'y2'})
        fig.layout.yaxis2.update({'title': 'Percentage'})

        # Update the margins to add a title and see graph x-labels.
        fig.layout.margin.update({'t': 75, 'l': 50})
        fig.layout.update({'title': f'Query #{q_id}: {all_results[(teams[0], q_id)][1]}'})

        # Update the height because adding a graph vertically will interact with
        # the plot height calculated for the table
        fig.layout.update({'height': 800})

        if not os.path.exists("images"):
            os.mkdir("images")
        # Plot!
        fig.write_image(f"./images/fig{q_id}.png") # TODO: Change the dest path

