from typing import List, Dict


def read_dataset(path: str, separator: str = ',', verbose: bool = False) -> Dict[str, List[float]]:
    if verbose:
        print(f'В файле по пути "{path}" ожидается разделитель `{separator}`')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except Exception as ex:
        raise ValueError(f"Мы не смогли открыть и прочитать файл по пути {path} ({ex})")
    if len(lines) < 2:
        raise ValueError("В файле меньше 2 строк, поэтому либо файл некорректен, либо в нём нет данных")
    header = lines[0].split(separator)
    if len(set(header)) < len(header):
        raise ValueError(f"В header-е есть одинаковые названия столбцов")
    header_length = len(header)
    dct: Dict[str, List[float]] = {}
    index_to_column: Dict[int, str] = {}
    for i, column in enumerate(header):
        index_to_column[i] = column
        dct[column] = []
    for i, line in enumerate(lines[1:], start=1):
        values = line.split(separator)
        if header_length != len(values):
            raise ValueError(f"Количество значений в header-е и в строке {i} отличается. В header-е {header_length}, "
                             f"а в строке {i} {len(values)}")
        try:
            values = list(map(float, values))
        except ValueError as ex:
            raise ValueError(f"Все значения в строке {i} невозможно привести к типу float ({ex})")
        for j, value in enumerate(values):
            dct[index_to_column[j]].append(value)
    return dct
