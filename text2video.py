import gradio
import towhee
import pandas as pd
import os

raw_video_path = './test_1k_compress' # 1k test video path.
test_csv_path = './MSRVTT_JSFUSION_test.csv' # 1k video caption csv.

test_sample_csv_path = './MSRVTT_JSFUSION_test_sample.csv'
device = 'cuda:0'
show_num = 3

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

sample_num = 1000 # you can change this sample_num to be smaller, so that this notebook will be faster.
test_df = pd.read_csv(test_csv_path)
print('length of all test set is {}'.format(len(test_df)))
sample_df = test_df.sample(sample_num, random_state=42)

sample_df['video_path'] = sample_df.apply(lambda x:os.path.join(raw_video_path, x['video_id']) + '.mp4', axis=1)

sample_df.to_csv(test_sample_csv_path)
print('random sample {} examples'.format(sample_num))

connections.connect(host='127.0.0.1', port='19530')

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
    FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='video retrieval')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2', #IP
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

collection = create_milvus_collection('text_video_retrieval', 512)

dc = (
    towhee.read_csv(test_sample_csv_path)
      .runas_op['video_id', 'id'](func=lambda x: int(x[-4:]))
      .video_decode.ffmpeg['video_path', 'frames'](sample_type='uniform_temporal_subsample', args={'num_samples': 12})
      .runas_op['frames', 'frames'](func=lambda x: [y for y in x])
      .video_text_embedding.clip4clip['frames', 'vec'](model_name='clip_vit_b32', modality='video', device=device)
      .to_milvus['id', 'vec'](collection=collection, batch=30)
)
print('Total number of inserted data is {}.'.format(collection.num_entities))

with towhee.api() as api:
    milvus_search_function = (
         api.clip4clip(model_name='clip_vit_b32', modality='text', device=device)
            .milvus_search(collection=collection, limit=show_num)
            .runas_op(func=lambda res: [os.path.join(raw_video_path, 'video' + str(x.id) + '.mp4') for x in res])
            .as_function()
    )

interface = gradio.Interface(milvus_search_function, 
                             inputs=[gradio.Textbox()],
                             outputs=[gradio.Video(format='mp4') for _ in range(show_num)]
                            )

interface.launch(inline=True, share=True)