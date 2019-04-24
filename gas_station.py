import sys
import sched
import time
import datetime
import pandas as pds
from bs4 import BeautifulSoup
from urllib import request

_TX_POOL_REQUEST = request.Request('https://ethgasstation.info/txPoolReport.php',
                                   headers={'User-Agent':
                                                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) '
                                                'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                'Chrome/73.0.3683.103 Safari/537.36'
                                            })
_PRED_TABLE_REQUEST = request.Request('https://ethgasstation.info/predictionTable.php',
                                      headers={'User-Agent':
                                                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) '
                                                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                   'Chrome/73.0.3683.103 Safari/537.36'
                                               }
                                      )


def get_tx_pool_report(req):
    with request.urlopen(req) as response:
        html = response.read()
    bs = BeautifulSoup(html, 'html.parser')
    table = bs.find(attrs={'class': 'x_content'}).find('table')
    df = pds.read_html(str(table))[0]
    block_number = int(bs.find(attrs={'class': 'x_title'}).find('span').text)

    df['timestamp'] = time.time()
    df['block'] = block_number
    return block_number, df


class RollingDataFrame:
    def __init__(self, file_name=None, req=_TX_POOL_REQUEST):
        if file_name is None:
            file_name = 'data/' + datetime.datetime.now().strftime('%y%m%d%H%M') + '-txpool.csv'
        self.file_name = file_name
        self.df = pds.DataFrame([])
        self.req = req
        self.last_block = -1
        self.last_saved = -1

    def add(self, block_number, df):
        if self.last_block < int(block_number):
            print('adding', block_number, len(df), 'entries', flush=True)
            self.df = self.df.append(df, ignore_index=True)
            if block_number > self.last_saved + 5:
                self.df.to_csv(self.file_name, index=False)
                self.last_saved = block_number
            self.last_block = block_number

    def run_forever(self, timeout=2):
        scheduler = sched.scheduler(time.time, time.sleep)

        def _worker():
            try:
                block, df = get_tx_pool_report(self.req)
                self.add(block, df)
            except Exception as e:
                print(e)
                print('Continue...')

            scheduler.enter(timeout, 1, _worker)

        scheduler.enter(0, 1, _worker)
        scheduler.run(blocking=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['txpool', 'predtable']:
        print('Usage: python gas_station.py txpool|predtable')
    if sys.argv[1] == 'txpool':
        name = 'data/' + datetime.datetime.now().strftime('%y%m%d%H%M') + '-txpool.csv'
        rdf = RollingDataFrame(name, _TX_POOL_REQUEST)
    elif sys.argv[1] == 'predtable':
        name = 'data/' + datetime.datetime.now().strftime('%y%m%d%H%M') + '-pred_table.csv'
        rdf = RollingDataFrame(name, _PRED_TABLE_REQUEST)
    else:
        raise Exception('Wrong use')
    rdf.run_forever()


if __name__ == '__main__':
    main()
