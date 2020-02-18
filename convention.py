import pandas as pd

class Convention:
    
    def __init__(self):
        self._delegate_allocation = None
    
    @property
    def delegate_allocation(self):
        if self._delegate_allocation is None:
            """read from original source"""
            df = pd.read_html('http://www.thegreenpapers.com/P20/D-Del.phtml', skiprows=1, header=0)[0]
            df.columns = df.columns.str.replace('\(sort\)','')
            mapper = {
                'Pledged PLEOs': r'(\d+) Pledged PLEO',
                'Unpledged PLEOs': r'(\d+) Unpledged PLEO',
                'District': r'([\d]+) district',
                'At Large': r'(\d+) at large'
             }
            for new_col, regex in mapper.items():
                df[new_col] = self.extract_del_count(df['Details of Allocation'], regex)
            df['notes'] = ["must be DNC members" if x == 'Unassigned' else '' for x in df.State]
            df = df.drop(columns=['Unpledged','Details of Allocation','Rank']).fillna(0)
            self._delegate_allocation = df
        return self._delegate_allocation
    
    @staticmethod
    def extract_del_count(col, regex):
        """Helper method for parsing TheGreenPapers table"""
        return col.str.replace(',','').str.extract(regex).fillna(0).astype(int)