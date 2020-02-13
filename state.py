import math
import io
import requests
import lxml.html
import pandas as pd
import numpy as np

def allocate_delegates(candidate_votes, delegate_count, verbose=False, full_results=False):
    """Follow https://www.thegreenpapers.com/P20/D-Math.phtml algorithm. This excludes re-assignment of people supporting non-viable candidates"""
    unallocated_delegates = delegate_count
    # Step 1: Tabulate the percentage of the vote that each preference receives.
    candidate_pcts = candidate_votes / candidate_votes.sum()
    if verbose:
        print("Step 1: Preference percentages\n", candidate_pcts)
    # Step 2: Re-tabulate the percentage of the vote, to three decimals, received by each presidential preference 
    # excluding the votes of presidential preferences whose percentages in step 1 fall below 15%. 
    # Fine point: If no Presidential preference reaches a 15% threshold, the threshold is the half the percentage 
    # of the vote received by the front-runner [Delegate Selection Rules for the 2020 Democratic National Convention 
    # - Rule 14.F.]. The total vote is called the qualified vote.
    if candidate_pcts.max() < .15:
        threshold = candidate_pcts.max() / 2
    else:
        threshold = .15
    qualified_vote = candidate_votes[candidate_pcts > threshold].copy()
    qualified_pcts = qualified_vote / qualified_vote.sum()
    if verbose:
        print("Step 2: Qualified vote\n", qualified_pcts)
    # Step 3: Multiply the number of delegates to be allocated by the percentage received by each presidential 
    # preference in step 2. 
    del_calcs = qualified_pcts * delegate_count
    if verbose:
        print("Step 3: Delegate fractions\n", del_calcs)
    # Step 4: Delegates are allocated to each presidential preference based on the whole numbers that result 
    # from the multiplication in Step 3.
    delegates = np.floor(del_calcs).astype(int)
    unallocated_delegates -= delegates.sum()
    if verbose:
        print("Step 4: Initial delegate allocation\n", delegates)
    # Step 5: The remaining delegates, if any, are awarded in order of the highest fractional remainders in Step 3.
    del_fracs = (del_calcs - delegates).sort_values(ascending=False)
    if verbose:
        print("Step 5: Additional delegate allocation\n", del_fracs)
    i = 0
    while unallocated_delegates > 0:
        delegates[del_fracs.index[i]] += 1
        i += 1
        unallocated_delegates -= 1
    if full_results:
        return {'candidate_pcts': candidate_pcts, 
                'qualified_pcts': qualified_pcts, 
                'del_calcs': del_calcs, 
                'del_fracs': del_fracs, 
                'delegates': delegates}
    return delegates

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
        return col.str.replace(',','').str.extract(regex).fillna(0).astype(int)
    
class State:
    def __init__(self, pleos, at_large_dels, is_caucus=False):
        self._results = None
        self._state_dels = None
        self.is_caucus = is_caucus
        self.state_pleos = pleos
        self.state_at_large_dels = at_large_dels
        self.total_state_dels = pleos + at_large_dels
        
    @staticmethod
    def viability_threshold(caucus_size, cutoff=.15):
        return math.ceil(caucus_size * .15)

    @property
    def viable_sdes(self):
        return self.sdes[self.sdes > self.viability_threshold(self.sdes.sum())]

    def display_state_dels(self):
        if self.is_caucus:
            votes = self.sdes
            viable_votes = self.viable_sdes
        else:
            votes = self.votes
            viable_votes = self.viable_votes
        pleos = allocate_delegates(votes, self.state_pleos, full_results=True)
        atlarge = allocate_delegates(votes, self.state_at_large_dels, full_results=True)
        vote_pct = viable_votes / viable_votes.sum() * 100
        df = pd.concat([vote_pct, 
                        pleos['del_calcs'],
                        pleos['delegates'],
                        atlarge['del_calcs'],
                        atlarge['delegates'],
                        pleos['delegates'] + atlarge['delegates']],
                       axis=1)
        df.columns=['%','PLEO Delegates (unrounded)', 'PLEO Delegates','At-Large Delegates (unrounded)', 'At-Large Delegates', 'Delegates']
        df = df[df.Delegates > 0].sort_values(by='%',ascending=False)
        return df
    
        
    @property
    def state_dels(self):
        if self._state_dels is None:
            state_dels_full = self.display_state_dels()
            self._state_dels = state_dels_full.Delegates
        return self._state_dels

class Iowa(State):
    def __init__(self):
        super().__init__(pleos=5, at_large_dels=9, is_caucus=True)
        self._counties = None
        self._state_dels = None
    
    dist_dels = pd.DataFrame([[1, 7], [2, 7], [3, 8], [4, 5]], columns=(["District","Delegates"]))
    
    @property 
    def results(self):
        """return dataframe with caucus results"""
        if self._results is None:
            url = "https://results.thecaucuses.org"
            r = requests.get(url)

            root = lxml.html.parse(io.StringIO(r.text)).getroot()
            # Bennet, Biden, etc.
            head = root.find_class("thead")[0]
            header = [x.text for x in list(head.iterchildren())]

            # First Expression, Final Expression, SDE, ...
            subhead = root.find_class("sub-head")[0]
            subheader = [x.text for x in list(subhead.iterchildren())]

            columns = pd.MultiIndex.from_arrays([
                pd.Series(header).fillna(method='ffill'),
                pd.Series(subheader).fillna(method='ffill').fillna('')
            ], names=['candidate', 'round'])

            counties = root.find_class("precinct-county")
            county_names = [x[0].text for x in counties]
            counties_data = root.find_class("precinct-data")
            county = counties_data[0]
            rows = []

            for name, county in zip(county_names, counties_data):
                if len(county) > 1:
                    # satellites only have a total
                    county = county[:-1]

                for precinct in county:
                    # exclude total
                    rows.append((name,) + tuple(x.text for x in precinct))

            results = (
                pd.DataFrame(rows, columns=columns)
                  .set_index(['County', 'Precinct'])
                  .apply(pd.to_numeric)
            )
            self._results = results
        return self._results

    @property
    def counties(self):
        if self._counties is None:
            self._counties = pd.read_csv('iowa_counties_cd.txt', dtype={'DISTRICT': int}, sep='\t')
        return self._counties
    
    def display_results(self):
        results = self.results
        return pd.concat([results.loc[:,(slice(None),'First Expression')].sum().droplevel(1),
           (results.loc[:,(slice(None),'First Expression')].sum().droplevel(1) / results.loc[:,(slice(None),'First Expression')].sum().sum() * 100).round(1),
           results.loc[:,(slice(None),'Final Expression')].sum().droplevel(1),
           (results.loc[:,(slice(None),'Final Expression')].sum().droplevel(1) / results.loc[:,(slice(None),'Final Expression')].sum().sum() * 100).round(1),
           results.loc[:,(slice(None),'SDE')].sum().droplevel(1).astype(int),
           (results.loc[:,(slice(None),'SDE')].sum().droplevel(1) / results.loc[:,(slice(None),'SDE')].sum().sum() * 100).round(1)], 
          axis=1).rename(columns={0:'First',1:'%',2:'Final',3:'%',4:'Total S.D.E.s',5:'%'}).sort_values(by='Final', ascending=False)
  
    @property
    def sdes(self):
        return self.results.loc[:,(slice(None),'SDE')].sum().droplevel(1).sort_values(ascending=False)


    def sum_dist_sdes(self, district, rounding=True):
        dist_counties = self.counties[self.counties.DISTRICT == district].COUNTY
        dist_sdes = self.results[self.results.index.isin(dist_counties, level='County')].loc[:,(slice(None),'SDE')]
        if rounding:
            return dist_sdes.sum().droplevel(1).round().astype(int)
        return dist_sdes.sum().droplevel(1)
    
    def display_dist_sdes(self, rounding=True, viable=False):
        all_dist_sdes = [self.sum_dist_sdes(i, rounding) for i in range(1, 5)]
        if viable:
            all_dist_sdes = [dist_sdes[dist_sdes > self.viability_threshold(dist_sdes.sum())] for dist_sdes in all_dist_sdes]
        all_dist_sdes = pd.concat(all_dist_sdes, axis=1, sort=True).sort_values(by=0, ascending=False).rename(columns={3:4,2:3,1:2,0:1})

        return all_dist_sdes
    
    @property
    def dist_sdes(self):
        return self.display_dist_sdes(rounding=False)
    
    def viable_dist_sdes(self):
        return self.dist_sdes[self.dist_sdes > self.viability_threshold(self.dist_sdes.sum())]
    
    # dist_pct = dist_sdes.div(dist_sdes.sum(axis=0), axis=1)
    
    def allocate_dist_dels(dist):
#        df = pd.concat([dist_pct[[dist]], dist_pct[[dist]] > self.viability_threshold(self.dist_sdes[, 
#                    (dist_pct[[dist]] * dist_delegates.query(f'District == {dist}').Delegates.tolist()[0]).round().astype(int)], 
#                   axis=1)
#        df.columns=['%','Viable','Delegates']
#        return df
        pass