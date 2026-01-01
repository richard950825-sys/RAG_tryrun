import os
# 必须放在最前面
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("⏳ 正在通过镜像站预下载 Docling 模型，请稍候...")

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

# 初始化转换器会触发模型下载
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# 随便转换一个空文件或者不存在的文件，目的是触发 artifacts 下载
try:
    print("🚀 开始触发下载流程...")
    # 这里不需要真正转换成功，只要 pipeline 初始化成功，模型就下来了
    converter.convert("https://arxiv.org/pdf/2206.01062") 
except Exception as e:
    print(f"⚠️ 转换过程报错(正常现象，只要模型下载了就行): {e}")

print("✅ 模型缓存检查完毕！")