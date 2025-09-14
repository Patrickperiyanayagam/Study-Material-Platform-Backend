from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
import os
import uuid
from app.models.schemas import DocumentUploadResponse
from app.services.document_processor import DocumentProcessor
# Environment variables will be accessed directly with os.getenv()

router = APIRouter()

def get_document_processor():
    return DocumentProcessor()

@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    print("=" * 60)
    print("📁 DOCUMENT UPLOAD API CALLED")
    print(f"📄 Files: {[file.filename for file in files]}")
    print(f"📊 Total files: {len(files)}")
    
    if not files:
        print("❌ No files provided")
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_files = []
    total_chunks = 0
    
    for file in files:
        print(f"   Processing: {file.filename} (size: {file.size} bytes)")
        
        max_file_size = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB default
        if file.size > max_file_size:
            print(f"❌ File too large: {file.filename}")
            raise HTTPException(
                status_code=413, 
                detail=f"File {file.filename} exceeds maximum size of {max_file_size} bytes"
            )
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        upload_dir = os.getenv("UPLOAD_DIR", "./data/documents")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            print(f"⚡ Processing document: {file.filename}")
            # Process document and add to vector store
            chunks = await processor.process_document(file_path, file.filename)
            total_chunks += len(chunks)
            processed_files.append(file.filename)
            print(f"✅ Processed {len(chunks)} chunks from {file.filename}")
        except Exception as e:
            print(f"❌ Failed to process {file.filename}: {str(e)}")
            # Clean up file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {str(e)}")
    
    print(f"✅ All documents processed successfully")
    print(f"📤 Total processed: {len(processed_files)} files, {total_chunks} chunks")
    print("=" * 60)
    
    return DocumentUploadResponse(
        message="Documents uploaded and processed successfully",
        file_count=len(processed_files),
        processed_chunks=total_chunks,
        file_names=processed_files
    )

@router.get("/status")
async def upload_status():
    return {"status": "Upload service is running"}

@router.delete("/documents")
async def clear_documents(processor: DocumentProcessor = Depends(get_document_processor)):
    try:
        await processor.clear_vector_store()
        
        # Clean up uploaded files
        upload_dir = os.getenv("UPLOAD_DIR", "./data/documents")
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")