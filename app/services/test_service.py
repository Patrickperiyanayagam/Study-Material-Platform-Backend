import asyncio
import json
import re
import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.document_processor import DocumentProcessor
from app.services.model_factory import ModelFactory
from app.models.schemas import TestQuestion, AnswerSubmission, QuestionGrade, GradingResponse

class TestService:
    def __init__(self, initial_config: Dict[str, Any] = None):
        self.document_processor = DocumentProcessor()
        
        # Load initial configuration or use defaults
        if initial_config:
            config = initial_config
        else:
            config = {
                "provider": "ollama",
                "model_name": "llama3.2",
                "temperature": 0.7,
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            }
        
        # Create LLM instance using factory
        self.current_config = config
        self.llm = self._create_llm_instance(config)
        self._initialize_prompts()
    
    def _create_llm_instance(self, config: Dict[str, Any]):
        """Create LLM instance from configuration."""
        try:
            print(f"🧪 TEST SERVICE - Creating LLM instance with config: {config}")
            
            api_key = None
            if config["provider"] != "ollama":
                api_key = ModelFactory.get_api_key_for_provider(config["provider"])
            
            print(f"🧪 TEST SERVICE - Calling ModelFactory.create_llm with:")
            print(f"   provider: {config['provider']}")
            print(f"   model_name: {config['model_name']}")
            print(f"   temperature: {config.get('temperature', 0.7)}")
            print(f"   base_url: {config.get('base_url')}")
            print(f"   api_key: {'<present>' if api_key else '<none>'}")
            print(f"   max_tokens: {config.get('max_tokens')}")
            
            return ModelFactory.create_llm(
                provider=config["provider"],
                model_name=config["model_name"],
                temperature=config.get("temperature", 0.7),
                base_url=config.get("base_url"),
                api_key=api_key,
                max_tokens=config.get("max_tokens")
            )
        except Exception as e:
            # Fallback to default Ollama configuration
            print(f"❌ Failed to create LLM with config {config}: {e}")
            print(f"🔄 Falling back to default Ollama configuration")
            return ModelFactory.create_llm(
                provider="ollama",
                model_name="llama3.2",
                temperature=0.7,
                base_url="http://localhost:11434"
            )
    
    def update_model_config(self, provider: str, model_name: str, temperature: float, 
                           base_url: str = None, api_key: str = None, max_tokens: int = None):
        """Update the model configuration and recreate the LLM instance."""
        try:
            print(f"🔧 TEST SERVICE - Updating model config to {provider}/{model_name}")
            
            config = {
                "provider": provider,
                "model_name": model_name,
                "temperature": temperature,
                "base_url": base_url,
                "api_key": api_key,
                "max_tokens": max_tokens
            }
            
            self.current_config = config
            self.llm = self._create_llm_instance(config)
            print(f"✅ Test service updated to use {provider} with model {model_name}")
                
        except Exception as e:
            print(f"❌ Failed to update test service model: {str(e)}")
            raise Exception(f"Failed to update model configuration: {str(e)}")
    
    def _initialize_prompts(self):
        """Initialize prompts for test generation and grading."""
        # Test generation prompt
        self.test_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert exam generator. Generate essay/short answer questions based on the provided context from documents.

Instructions:
1. Create {num_questions} questions with the specified mark distribution
2. Make questions test deep understanding, analysis, and application of concepts
3. Difficulty level: {difficulty}
4. Mark allocation:
   - 2 marks: Basic recall and understanding questions
   - 4 marks: Application and analysis questions  
   - 8 marks: Evaluation, synthesis, and complex analysis questions
5. Ensure questions are syllabus-aligned and test different aspects of the material

Mark Distribution: {mark_distribution}

Context from documents:
{context}

Return the questions in the following JSON format:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "marks": 2,
      "difficulty": "easy",
      "topic": "Main topic/concept being tested"
    }}
  ]
}}"""),
            ("human", "Generate {num_questions} questions with {difficulty} difficulty and the specified mark distribution based on the provided context.")
        ])
        
        # Grading prompt
        self.grading_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert exam grader with extensive knowledge in the subject area. Grade the student's answer based on the marking rubric and provided context.

Question: {question}
Maximum Marks: {max_marks}
Student Answer: {student_answer}

Reference Context:
{context}

Grading Instructions:
1. Assess the answer against the reference context and expected knowledge level
2. Award marks based on:
   - Accuracy of information (40%)
   - Depth of understanding (30%) 
   - Use of relevant examples/evidence (20%)
   - Clarity of explanation (10%)
3. Provide specific, constructive feedback
4. Identify strengths and areas for improvement
5. Be fair but rigorous in marking

Return your assessment in the following JSON format:
{{
  "marks_awarded": 0.0,
  "percentage": 0.0,
  "feedback": "Detailed feedback on the answer",
  "strengths": ["List of strong points in the answer"],
  "improvements": ["List of areas that need improvement"]
}}"""),
            ("human", "Grade this student answer based on the question, context, and rubric. Provide detailed feedback.")
        ])
    
    async def generate_test(
        self, 
        num_questions: int = 10,
        difficulty: str = "medium",
        mark_distribution: Dict[int, int] = None,
        topics: Optional[List[str]] = None
    ) -> List[TestQuestion]:
        """Generate test questions from the document corpus."""
        try:
            if mark_distribution is None:
                mark_distribution = {2: 5, 4: 3, 8: 2}
            
            # Get relevant content from documents
            if topics:
                # If specific topics are requested, search for them
                all_content = []
                for topic in topics:
                    docs = await self.document_processor.search_documents(topic, k=5)
                    all_content.extend(docs)
            else:
                # Get diverse content from all documents
                sample_queries = [
                    "main concepts", "key theories", "important principles",
                    "definitions", "examples", "case studies", "methods",
                    "processes", "applications", "analysis", "evaluation"
                ]
                all_content = []
                for query in sample_queries:
                    docs = await self.document_processor.search_documents(query, k=3)
                    all_content.extend(docs)
            
            if not all_content:
                raise Exception("No content available for test generation. Please upload documents first.")
            
            # Format context
            context = self._format_context_for_test(all_content, num_questions)
            
            # Generate test using LLM
            chain = self.test_prompt | self.llm | StrOutputParser()
            
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "num_questions": num_questions,
                    "difficulty": difficulty,
                    "mark_distribution": mark_distribution,
                    "context": context
                }
            )
            
            # Parse the JSON response
            test_data = self._parse_test_response(response)
            
            # Convert to TestQuestion objects and ensure mark distribution
            questions = []
            mark_counts = {2: 0, 4: 0, 8: 0}
            
            for q_data in test_data.get("questions", []):
                try:
                    marks = int(q_data.get("marks", 2))
                    if marks not in [2, 4, 8]:
                        marks = 2  # Default to 2 marks
                    
                    # Check if we still need this mark category
                    if mark_counts.get(marks, 0) < mark_distribution.get(marks, 0):
                        question = TestQuestion(
                            question=q_data["question"],
                            marks=marks,
                            difficulty=q_data.get("difficulty", difficulty),
                            topic=q_data.get("topic", "General")
                        )
                        questions.append(question)
                        mark_counts[marks] += 1
                except (KeyError, ValueError) as e:
                    # Skip malformed questions
                    continue
            
            # Fill remaining questions if needed
            total_needed = sum(mark_distribution.values())
            while len(questions) < total_needed and len(questions) < num_questions:
                # Add basic questions if we don't have enough
                for marks in [2, 4, 8]:
                    if mark_counts.get(marks, 0) < mark_distribution.get(marks, 0):
                        question = TestQuestion(
                            question=f"Explain the key concepts related to the main topics covered in the material. ({marks} marks)",
                            marks=marks,
                            difficulty=difficulty,
                            topic="General"
                        )
                        questions.append(question)
                        mark_counts[marks] += 1
                        break
                else:
                    break
            
            if not questions:
                raise Exception("Failed to generate valid test questions")
            
            return questions[:num_questions]
            
        except Exception as e:
            raise Exception(f"Test generation failed: {str(e)}")
    
    async def grade_test(
        self,
        questions: List[TestQuestion],
        answers: List[AnswerSubmission]
    ) -> GradingResponse:
        """Grade student answers for the test."""
        try:
            grades = []
            total_marks_awarded = 0.0
            total_marks_possible = 0
            weak_topics = []
            
            # Create answer lookup for efficient access
            answer_lookup = {ans.question_index: ans.answer for ans in answers}
            
            for i, question in enumerate(questions):
                total_marks_possible += question.marks
                user_answer = answer_lookup.get(i, "")
                
                if not user_answer.strip():
                    # Empty answer
                    grade = QuestionGrade(
                        question_index=i,
                        question=question.question,
                        user_answer=user_answer,
                        marks_awarded=0.0,
                        max_marks=question.marks,
                        percentage=0.0,
                        feedback="No answer provided.",
                        strengths=[],
                        improvements=["Please provide an answer to demonstrate your understanding."]
                    )
                else:
                    # Grade the answer using AI
                    grade = await self._grade_single_answer(question, user_answer, i)
                
                grades.append(grade)
                total_marks_awarded += grade.marks_awarded
                
                # Track weak topics (less than 50% score)
                if grade.percentage < 50:
                    weak_topics.append(question.topic)
            
            # Calculate overall percentage
            overall_percentage = (total_marks_awarded / total_marks_possible * 100) if total_marks_possible > 0 else 0
            
            # Generate overall feedback and study plan
            overall_feedback = self._generate_overall_feedback(overall_percentage, grades)
            study_plan = self._generate_study_plan(weak_topics, overall_percentage)
            
            return GradingResponse(
                grades=grades,
                total_marks_awarded=total_marks_awarded,
                total_marks_possible=total_marks_possible,
                overall_percentage=overall_percentage,
                overall_feedback=overall_feedback,
                weak_topics=list(set(weak_topics)),  # Remove duplicates
                study_plan=study_plan
            )
            
        except Exception as e:
            raise Exception(f"Test grading failed: {str(e)}")
    
    async def _grade_single_answer(
        self,
        question: TestQuestion,
        user_answer: str,
        question_index: int
    ) -> QuestionGrade:
        """Grade a single answer using AI."""
        try:
            # Get relevant context for this question
            context_docs = await self.document_processor.search_documents(
                question.question, k=2
            )
            context = self._format_context_for_grading(context_docs)
            
            # Grade using LLM
            chain = self.grading_prompt | self.llm | StrOutputParser()
            
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "question": question.question,
                    "max_marks": question.marks,
                    "student_answer": user_answer,
                    "context": context
                }
            )
            
            # Parse grading response
            grading_data = self._parse_grading_response(response)
            
            marks_awarded = min(float(grading_data.get("marks_awarded", 0)), question.marks)
            percentage = (marks_awarded / question.marks * 100) if question.marks > 0 else 0
            
            return QuestionGrade(
                question_index=question_index,
                question=question.question,
                user_answer=user_answer,
                marks_awarded=marks_awarded,
                max_marks=question.marks,
                percentage=percentage,
                feedback=grading_data.get("feedback", ""),
                strengths=grading_data.get("strengths", []),
                improvements=grading_data.get("improvements", [])
            )
            
        except Exception as e:
            # Fallback grading
            return QuestionGrade(
                question_index=question_index,
                question=question.question,
                user_answer=user_answer,
                marks_awarded=question.marks * 0.5,  # Give 50% as fallback
                max_marks=question.marks,
                percentage=50.0,
                feedback=f"Automatic grading unavailable. Please review manually. Error: {str(e)}",
                strengths=["Answer provided"],
                improvements=["Please review with instructor for detailed feedback"]
            )
    
    def _format_context_for_test(self, documents: List[Dict], num_questions: int) -> str:
        """Format documents into context suitable for test generation."""
        # Remove duplicates and get diverse content
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content = doc["content"]
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)
        
        # Limit context length to avoid token limits
        context_parts = []
        total_length = 0
        max_length = 6000  # Increased for test generation
        
        for doc in unique_docs:
            source = doc["metadata"].get("source", "Document")
            content = doc["content"]
            
            if total_length + len(content) < max_length:
                context_parts.append(f"Source: {source}\nContent: {content}")
                total_length += len(content)
            else:
                break
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_context_for_grading(self, documents: List[Dict]) -> str:
        """Format documents into context suitable for grading."""
        if not documents:
            return "No reference context available."
        
        context_parts = []
        for doc in documents[:3]:  # Limit to top 3 most relevant
            source = doc["metadata"].get("source", "Document")
            content = doc["content"][:1000]  # Limit length per document
            context_parts.append(f"Source: {source}\nContent: {content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _parse_test_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for test generation."""
        try:
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError:
            # Fallback: create simple questions
            return {
                "questions": [
                    {
                        "question": "Explain the key concepts from the provided material.",
                        "marks": 4,
                        "difficulty": "medium",
                        "topic": "General"
                    }
                ]
            }
    
    def _parse_grading_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for grading."""
        try:
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError:
            # Fallback grading
            return {
                "marks_awarded": 0.0,
                "percentage": 0.0,
                "feedback": "Unable to parse grading response.",
                "strengths": [],
                "improvements": ["Please review manually"]
            }
    
    def _generate_overall_feedback(self, percentage: float, grades: List[QuestionGrade]) -> str:
        """Generate overall feedback based on performance."""
        if percentage >= 85:
            return "Excellent work! You demonstrate strong understanding across all topics."
        elif percentage >= 70:
            return "Good performance overall. Focus on the weaker areas for improvement."
        elif percentage >= 50:
            return "Satisfactory performance. Significant improvement needed in several areas."
        else:
            return "Additional study and practice required. Consider reviewing the fundamental concepts."
    
    def _generate_study_plan(self, weak_topics: List[str], percentage: float) -> List[str]:
        """Generate study plan based on weak topics."""
        plan = []
        
        if weak_topics:
            plan.append(f"Review fundamental concepts in: {', '.join(set(weak_topics))}")
            plan.append("Practice additional exercises on weak topics")
        
        if percentage < 70:
            plan.append("Revisit course materials and take detailed notes")
            plan.append("Seek additional help from instructors or study groups")
        
        if percentage < 50:
            plan.append("Start with basic concepts before advancing to complex topics")
            plan.append("Consider additional study resources and practice tests")
        
        if not plan:
            plan.append("Continue practicing to maintain your excellent performance")
        
        return plan
    
    async def get_available_topics(self) -> List[str]:
        """Get available topics from uploaded documents."""
        try:
            # This would ideally extract topics from document metadata or content
            # For now, return some common academic topics
            return [
                "Introduction", "Fundamentals", "Core Concepts",
                "Applications", "Case Studies", "Analysis",
                "Theory", "Practice", "Implementation"
            ]
        except Exception:
            return ["General"]