import pandas as pd 

if __name__ == '__main__':
    # read in the data
    df = pd.read_csv('data/orig_data.txt')

    # use mean normalization
    def normalize(data_subset):
        # normalize only the 11 proteins
        data_subset.loc[:, 'raf':'jnk'] = data_subset.loc[:, 'raf':'jnk'].apply(lambda x: (x - x.mean()) / x.std())
        return data_subset
    
    # perform stratified normalization
    indices = [-1, 852, 1754, 2664, 3388, 4198, 4997, 5845, 6758, 7465]
    for i in range(1, len(indices)):
        start, end =  indices[i-1] + 1, indices[i]
        df.loc[start:end] = normalize(df.loc[start:end])

    # # write to new txt
    df.to_csv('data/processed_data.txt', index=False)
    