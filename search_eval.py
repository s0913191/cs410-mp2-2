import math
import sys
import time
import metapy
import pytoml


class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """

    def __init__(self, some_param=1.0):
        self.param = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        doc_size = sd.doc_size
        doc_term_count = sd.doc_term_count
        corpus_term_count = sd.corpus_term_count
        query_term_weight = sd.query_term_weight
        N = sd.num_docs
        avg_dl = sd.avg_dl
        c = self.param

        tfn = doc_term_count * math.log(1 + avg_dl/doc_size,2)
        score = query_term_weight*(tfn/(tfn+c))*math.log((N+1)/(corpus_term_count+0.5),2)
        return score


def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """
    
    #Score:0.3907125886476737
    #return InL2Ranker()

    #Score: 0.387519660757908
    #return InL2Ranker(5)

    #Score: 
    #return metapy.index.DirichletPrior(1000)

    #Score: 0.295647752245879
    #return metapy.index.DirichletPrior(3000)

    #Score: 0.30931006020423013
    #return metapy.index.DirichletPrior()

    #Score: 0.35572441040983604
    #return metapy.index.JelinekMercer()

    #Score: 0.3820866842006129
    #return metapy.index.AbsoluteDiscount()

    #Score: 0.4091226999465542
    #return metapy.index.PivotedLength()

    #Score: 0.41435984732626685
    #return metapy.index.PivotedLength(0.15)

    #Score: 0.4170671141624618
    #return metapy.index.OkapiBM25(k1=1.2,b=0.75,k3=500)

    #PASSED!! Score: 0.42159329113868105
    #return metapy.index.OkapiBM25(k1=2,b=0.75,k3=1000)

    #Score: 0.36426628425723906	
    #return metapy.index.OkapiBM25(k1=2,b=1,k3=6.5)

    # 0.4216823050552537
    #return metapy.index.OkapiBM25(k1=1.8,b=0.75,k3=6.5)


    return metapy.index.OkapiBM25(k1=2,b=0.6,k3=6.5)

    #SOTA 0.42763566925866947
    #return metapy.index.OkapiBM25(k1=2,b=0.7,k3=6.5)

    #PASSED!! Score: 0.42222875978595253
    #return metapy.index.OkapiBM25(k1=2,b=0.75,k3=6.5)

    #PASSED!! Score: 0.42197778514647183
    #return metapy.index.OkapiBM25(k1=2,b=0.75,k3=10)

    #PASSED!! Score: 0.4208391018854435
    #return metapy.index.OkapiBM25(k1=2.5,b=0.75,k3=1000)

    #Score: 0.3673030717237078
    #return metapy.index.OkapiBM25(k1=1.5,b=1,k3=500)

    #Score: 0.37019729527593204
    #return metapy.index.OkapiBM25(k1=1,b=1,k3=500)

    #Score: 0.418009081473077
    #return metapy.index.OkapiBM25() 

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()
    ndcg = 0.0
    num_queries = 0

    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            ndcg += ev.ndcg(results, query_start + query_num, top_k)
            num_queries+=1
    ndcg= ndcg / num_queries
            
    print("NDCG@{}: {}".format(top_k, ndcg))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
