## 쿼리 결과물 출력 함수
def print_result(text: str, results: tuple):
    
    print(f'[{text}]')
    for result in results: print(result)
    print('\n')
    
    
## 조회 함수
def select(cursor, text : str, table_name: str, column: str = '*',
           order: str = None, cond: str = None, limit_k: str = None, 
           group: str = None, is_print: bool = True
    ):

    query    = f'select {column} from {table_name}'
    
    if    cond: query += f' where {cond}'
    if   group: query += f' group by {group}'
    if   order: query += f' order by {order}'
    if limit_k: query += f' limit {limit_k}'

    print(f'[query] {query};')
    cursor.execute(query)
    
    if is_print: print_result(text, cursor.fetchall())


## Row 추가 함수
def insert_(cursor, table_name: str, values: str, 
           column = None, copy = None):

    column = f'({column}) ' if column else ''
    query  = f'insert into {table_name} {column}'

    if copy:
        query += copy
        
    else:
        query += 'values'
        for idx, value in enumerate(values, 1): 
    
            value = ', '.join(value)
            query += f' ({value})' if idx == len(values) else f' ({value}),'
            
            
    print(f'[query] {query};')


    try: 
        cursor.execute(query)
        print('[INFO] 데이터 삽입 완료 \n')
        
    except Exception as e:
        print(f'[ERROR] {e}')
        print('[ERROR] 쿼리에 문제가 발생하였습니다.')


## 테이블에 있는 레코드를 제거해주는 함수
def delete_(cursor, table_name: str, cond: str = None):

    query = f'delete from {table_name}'
    if cond: query += f' where {cond}'

    print(f'[query] {query};')
    print('[INFO] 데이터 제거 완료 \n')
    cursor.execute(query)


## 테이블에 있는 레코드를 갱신해주는 함수
def update_(cursor, table_name: str, set_: str, 
            cond: str = None):


    query = f'update {table_name} set {set_}'
    if cond: query += f' where {cond}'

    print(f'[query] {query};')
    print('[INFO] 데이터 업데이트 완료 \n')
    cursor.execute(query)

    
    






    

