import multiprocessing as mp
import logging, tqdm
import json, zlib, base64, compress_pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ZIPJSON_KEY = 'base64(zip(o))'

def json_row_zip(j_row):
    return base64.b64encode(
            zlib.compress(
                json.dumps(j_row).encode('utf-8')
                )
            ).decode('ascii')

def json_zip(j, total=None):
    """ Compresses json list of data.
        
        Expected format:
        
            data = [
                    {"id": 0, "feature_1": "level_0", "feature_2": "level_5"...},
                    ...,
                    {"id":100, "feature_1": "level_0", "feature_2": "level_5"...},
                    ]
    """
    # Initialize
    zip_j = {
        ZIPJSON_KEY: [],
        }

    # Determine good chunksize
    if total is None:
        chunksize = 1000
    else:
        chunksize = int(total / mp.cpu_count())
    with mp.Pool(mp.cpu_count()) as pool:
        zip_j[ZIPJSON_KEY] = list(pool.imap(
            json_row_zip,
            tqdm.tqdm(
                j,
                desc='Compressing',
                total=total,
                ),
            chunksize=chunksize,
            )
            )
    
    #for j_row in tqdm.tqdm(j, desc='Compressing'):
    #    zip_j[ZIPJSON_KEY].append(
    #            json_row_zip(j_row)
    #        )
    
    return zip_j

def json_unzip_generator(j, stop_idx):
    for i, compressed_row in enumerate(j):
        if stop_idx is not None:
            if i == stop_idx:
                break
        try:
            j_row = zlib.decompress(base64.b64decode(compressed_row))
        except:
            raise RuntimeError("Could not decode/unzip the contents")
        yield json.loads(j_row)

def json_unzip_all(j, verbose=False, stop_idx=None):
    _j = []
    for i, compressed_row in tqdm.tqdm(enumerate(j), desc='Loading row', leave=False, disable=not verbose):
        if stop_idx is not None:
            if i == stop_idx:
                break
        try:
            j_row = zlib.decompress(base64.b64decode(compressed_row))
        except:
            raise RuntimeError("Could not decode/unzip the contents")
        try:
            _j.append(json.loads(j_row))
        except:
            raise RuntimeError("Could interpret the unzipped contents")
    return _j


def json_unzip(j, insist=True, as_generator=True, verbose=False, stop_idx=None):
    """ Unzips compressed json.
        
        Expected format:
        
            data = [
                    {"id": 0, "feature_1": "level_0", "feature_2": "level_5"...},
                    ...,
                    {"id":100, "feature_1": "level_0", "feature_2": "level_5"...},
                    ]
    """
    try:
        assert (j[ZIPJSON_KEY])
        assert (set(j.keys()) == {ZIPJSON_KEY})
    except:
        if insist:
            raise RuntimeError("JSON not in the expected format {" + str(ZIPJSON_KEY) + ": zipstring}")
        else:
            return j

    j = j[ZIPJSON_KEY]

    if as_generator:
        return json_unzip_generator(j, stop_idx)
    else:
        return json_unzip_all(j, verbose=verbose, stop_idx=stop_idx)

def big_dict_generator(big_dict):
    for row in big_dict:
        yield row

class JsonDataStream:
    def __init__(self, filepath=None, json_data=None, compression='lz4'):
        self.filepath = filepath
        self.json_data = json_data
        self.total = None
        self.compression = compression
        self._setup()

    def _setup(self):
        if self.filepath is not None:
            with open(self.filepath, 'rb') as f_:
                compressed_data = compress_pickle.load(f_, compression=self.compression)
            self.data_gen = json_unzip(compressed_data)
        elif self.json_data is not None:
            # Build into its own iterator
            self.data_gen = big_dict_generator(self.json_data)
            self.total = len(self.json_data)
        else:
            raise ValueError('Need to specify filepath to data or json data.')

    def save(self, filepath, compression=None):
        if compression is None:
            compression = self.compression
        with open(filepath, 'wb') as f_:
            compress_pickle.dump(
                    json_zip(
                        self.data_gen,
                        ),
                    f_,
                    compression=compression,
                    )

    def __next__(self):
        return next(self.data_gen)

    def __iter__(self):
        return self.data_gen

def print_id(data):
    print(data["id"])


if __name__ == "__main__":
    import pickle

    with open('/home/public/data/ncats/correlation_study/raw_staged_data/discretized/discretized_expr_fpkm_uq_corr_data', 'rb') as f_:
        df = pickle.load(f_)
    logger.info('Loaded pickled data')
    dataset = df.to_dict(orient='index')
    logger.info('Casted pandas DF into dictionary.') 
    # Convert dataset
    new_dataset = []
    for example, feature_dict in tqdm.tqdm(dataset.items(), desc='Processing example', total=len(dataset)):
        data_row = {"id": example}
        data_row.update(feature_dict)
        new_dataset.append(data_row)

    stream = JsonDataStream(json_data=new_dataset)
    #print(next(stream)["id"])
    logger.info('Saving stream.')
    stream.save('discretized_expr_fpkm_uq_corr_data.dat')
    '''
    stream = JsonDataStream(filepath='discretized_expr_fpkm_corr_data.dat')
    with mp.Pool(5) as pool:
        res = list(
                pool.imap(
                    print_id, 
                    stream,
                    chunksize=1,
                    )
                )
    '''
