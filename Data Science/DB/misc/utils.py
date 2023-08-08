## 쿼리 결과물 출력 함수
def print_result(text: str, results: tuple):
    
    print(f'[{text}]')
    for result in results: print(result)
    print('\n')
    
    
## 조회 함수
def select(cursor, text : str, table_name: str, column: str = '*',
           order: str = None, cond: str = None, limit_k: str = None, 
           is_print: bool = True
    ):

    query    = f'select {column} from {table_name}'
    
    if  cond: query   += f' where {cond}'
    if order: query   += f' order by {order}'
    if limit_k: query += f' limit {limit_k}'
    
    print(f'[query] {query};\n')
    cursor.execute(query)
    
    if is_print: print_result(text, cursor.fetchall())

    
    
