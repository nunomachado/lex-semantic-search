import re
import pandas as pd


def is_valid_sentence(text):
    # check if text starts with a capitalized letter and ends with a punctuation mark
    return re.match(r"^[A-Z].*", text) and re.match(r".*[.?!]\s*$", text)


def group_by_sentences(df):
    """Group episode transcript passages by sentences.
    """
    episode_id = None
    guest = None
    title = None
    text = ""
    start = None
    end = None
    new_rows = []

    for i, row in df.iterrows():
        # continue previous sentence that wasn't complete
        if episode_id == row["id"]:
            # append the current text to the previous text and merge timestamps
            text += " " + row["text"]
            end = row["end"]
        else:
            # otherwise, create a new row and reset variables
            episode_id = row["id"]
            guest = row["guest"]
            title = row["title"]
            text = row["text"]
            start = row["start"]
            end = row["end"]

        if is_valid_sentence(text):
            # add new sentence if valid and reset id
            new_rows.append([episode_id, guest, title, text, start, end])
            episode_id = None

    # add the last row to the new_rows list
    # new_rows.append([id, guest, title, text, start, end])

    # create a new dataframe with the new rows
    new_df = pd.DataFrame(new_rows, columns=df.columns)

    return new_df


def group_by_chunks(df, chunk_size=10):
    """Group episode transcript passages by chunks (where each chunk is a series of 'chunk_size' contiguous passages).
    """
    # create an empty DataFrame with the same columns as the input df
    chunked_df = pd.DataFrame(columns=df.columns)

    # group the dataframe by episodes
    grouped_df = df.groupby('id')

    # iterate over the groups
    for id_val, id_df in grouped_df:
        # iterate over the rows in chunks of size chunk_size
        for i in range(0, len(id_df), chunk_size):
            chunk = id_df.iloc[i:i+chunk_size]

            # concatenate the text values and update the start and end values for the current chunk
            text = " ".join(chunk['text'])
            start = chunk['start'].iat[0]
            end = chunk['end'].iat[-1]

            # create a new row for the current chunk with the concatenated text and updated start and end values
            new_row = pd.DataFrame({
                'id': [chunk['id'].iat[0]],
                'guest': [chunk['guest'].iat[0]],
                'title': [chunk['title'].iat[0]],
                'text': [text], 'start': [start],
                'end': [end]
            })

            # add the new row to the chunked_df DataFrame
            chunked_df = pd.concat([chunked_df, new_row], ignore_index=True)

    return chunked_df
