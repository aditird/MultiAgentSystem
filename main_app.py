################################
####                        ####
####  Multi Agent System    ####
####  main_app.py           ####            
####  ADITI R DESHPANDE     ####    
####                        ####
################################

import asyncio
from playwright.async_api import async_playwright
import json
import os
import base64
from datetime import datetime
from urllib.parse import urljoin
import re
import requests
from transformers import pipeline
import time
import random

class GoogleSignInHandler:
    """Handles Google sign-in with proper bot avoidance"""
    
    def __init__(self, page):
        self.page = page
        
    async def detect_google_signin(self):
        """Detect if Google sign-in is present on the page"""
        google_selectors = [
            'a[href*="accounts.google.com"]',
            'button[data-provider="google"]',
            'button:has-text("Google")',
            'button:has-text("Sign in with Google")',
            'button:has-text("Continue with Google")',
            '.google-signin',
            '[aria-label*="Google"]',
            'button[id*="google"]'
        ]
        
        for selector in google_selectors:
            elements = await self.page.query_selector_all(selector)
            if elements:
                print(f"🔍 Found Google sign-in with selector: {selector}")
                return True
        return False
    
    async def wait_for_manual_google_signin(self):
        """Wait for user to manually complete Google sign-in"""
        try:
            print("🔐 Google sign-in required (automation blocked)")
            print("   Please complete the Google sign-in manually in the browser window")
            print("   The system will wait for you to complete authentication...")
            
            # Store original URL to detect when we return to the app
            original_url = self.page.url
            
            # Wait for URL to change (user navigates to Google sign-in)
            print("   ⏳ Waiting for you to click Google sign-in...")
            await self.page.wait_for_event('framenavigated', timeout=120000)  # 2 minutes
            
            # Wait for user to complete sign-in and return to app
            print("   ✅ Detected navigation to Google sign-in")
            print("   ⏳ Please complete the sign-in process...")
            
            # Wait for navigation back to application or timeout
            try:
                # Wait for navigation back to a non-Google URL
                await self.page.wait_for_function(
                    """() => !window.location.href.includes('accounts.google.com')""",
                    timeout=300000  # 5 minutes
                )
                print("   ✅ Detected return to application")
                await self.page.wait_for_timeout(3000)  # Additional wait for page load
                return True
            except Exception as e:
                print(f"   ⚠️  Still waiting for sign-in completion: {e}")
                # Fallback: wait for significant time and check if we're back
                await self.page.wait_for_timeout(30000)  # 30 seconds
                current_url = self.page.url
                if 'accounts.google.com' not in current_url:
                    print("   ✅ Returned to application (fallback detection)")
                    return True
                return False
                
        except Exception as e:
            print(f"❌ Error during manual sign-in wait: {e}")
            return False

class HuggingFaceDialoGPTAgent:
    """Agent that uses local model for question generation"""
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"  # Use smaller model
        self.generator = None
        self.conversation_history = []
        self.model_loaded = self.load_model()
    
    def load_model(self):
        """Load HuggingFace model with better error handling"""
        try:
            print(f"🔄 Loading HuggingFace model: {self.model_name}")
            
            # Use a smaller model and faster settings
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=-1,  # Use CPU
                torch_dtype="auto",  # Auto-detect data type
                max_length=200,  # Smaller max length
                truncation=True
            )
            
            print(f"✅ Successfully loaded model: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Using fallback question generation")
            self.generator = None
            return False

    def debug_conversation_history(self):
        """Debug method to print current conversation state"""
        print(f"🔍 Conversation History Debug:")
        print(f"   Total entries: {len(self.conversation_history)}")
        for i, conv in enumerate(self.conversation_history):
            question = conv.get('last_question', 'MISSING')
            response = conv.get('last_response', 'MISSING')
            print(f"   {i+1}. Q: {question[:60]}...")
            print(f"      A: {response[:60]}...")
            
    def generate_questions(self, goal, context=None, max_questions=1):
        """Generate questions using local model or fallback"""

        # Debug conversation history
        self.debug_conversation_history()
    
        if self.generator is None:
            print("🤖 Using rule-based questions (model not available)")
            return self._generate_fallback_questions(goal)
        
        try:
            # Prepare prompt
            if context and self.conversation_history:
                prompt = self._build_conversation_prompt(goal, context)
            else:
                prompt = self._build_initial_prompt(goal)
            
            print(f"📝 Prompt: {prompt[:100]}...")
            
            # Generate 
            response = self.generator(
                prompt,
                max_new_tokens=100,  # Use max_new_tokens instead of max_length
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                truncation=True
            )
            print("response = ", response)
            
            if response and len(response) > 0:
                generated_text = response[0]['generated_text']
                questions = self._extract_questions(generated_text, goal)
                
                if questions:
                    self.conversation_history.append({
                        "goal": goal,
                        "prompt": prompt,
                        "response": generated_text,
                        "questions": questions,
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"✅ AI generated {len(questions)} questions")
                    return questions[:max_questions]
            
            return self._generate_fallback_questions(goal)
            
        except Exception as e:
            print(f"⚠️ Generation failed: {e}")
            return self._generate_fallback_questions(goal)
    
    def _build_initial_prompt(self, goal):
        """Build initial prompt for DialoGPT"""

        return f"""I need to explore a website to find specific information about: "{goal}"

As an AI assistant helping someone navigate a website to accomplish a specific task, generate ONE specific, actionable question that will help find the exact information or functionality needed.

The question should focus on:
- Navigation to find specific sections/pages
- Locating application forms or procedures  
- Finding detailed information about processes
- Discovering where specific actions can be performed

Goal: {goal}

Generate exactly ONE specific navigation-focused question that will help accomplish this goal:"""
    


    def _build_conversation_prompt(self, goal, previous_context):
        """Build prompt with conversation context"""
        # Filter out empty conversations and get recent exchanges
        valid_conversations = [
            ctx for ctx in self.conversation_history 
            if ctx.get('last_question') and ctx.get('last_response')
        ]
        
        if not valid_conversations:
            print("📝 No valid conversation history found, using initial prompt")
            return self._build_initial_prompt(goal)
        
        # Get last 2 valid conversations
        recent_conversation = "\n".join([
            f"Previous Question: {ctx.get('last_question', '')}\nWhat we learned: {ctx.get('last_response', '')}" 
            for ctx in valid_conversations[-2:]
        ])
        
        #print(f"📝 Building conversation prompt with {len(valid_conversations)} previous exchanges")
        
        return f"""We're exploring a website to: "{goal}"

    Based on our previous exploration:
    {recent_conversation}

    Now generate exactly ONE follow-up question that will help us dig deeper to find the specific information or functionality we need. Focus on:

    - Navigating to more specific sections based on what we've learned
    - Finding the exact information needed for our goal
    - Locating forms, procedures, or detailed content
    - Discovering where the actual task can be completed

    Generate exactly ONE specific follow-up navigation question:"""

    
    def _extract_questions(self, generated_text, goal):
        """Extract clean questions from generated text"""
        # Split by common question markers
        #print("generated_text = ", generated_text)
        lines = generated_text.split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            # Look for lines that seem like questions
            if '?' in line or any(marker in line.lower() for marker in ['how', 'what', 'where', 'when', 'why', 'which', 'find', 'locate', 'navigate']):
                # Clean up the line - remove numbering, bullets, and prompt artifacts
                line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove numbering
                line = re.sub(r'^[-*]\s*', '', line)  # Remove bullets
                line = re.sub(r'^(question|follow.up|generate|goal).*?:', '', line, flags=re.IGNORECASE)  # Remove prompt prefixes
                line = line.strip()
                
                # Ensure it's a reasonable length and seems like a navigation/question
                if (15 <= len(line) <= 150 and 
                    ('?' in line or any(marker in line.lower() for marker in ['how', 'what', 'where', 'find', 'locate', 'navigate'])) and
                    not line.lower().startswith('goal:') and
                    not line.lower().startswith('generate')):
                    questions.append(line)
        
        # If no good questions found, use fallback
        print("questions = ", questions)
        if not questions:
            return self._generate_fallback_questions(goal)
    
        return questions


    def _generate_fallback_questions(self, goal):
        """Generate fallback questions when API is unavailable"""
        goal_lower = goal.lower()
        #print("came into generate_fallback_questions")
##        pass
        
        # Context-aware fallback questions
        if any(word in goal_lower for word in ['apply', 'application', 'admission', 'enroll']):
            return [
                "Where can I find the admissions or application section?",
                "How do I navigate to the application portal or form?",
                "Where are the step-by-step application instructions?",
                "What is the direct link to start the application process?"
            ]
        elif any(word in goal_lower for word in ['program', 'degree', 'course', 'major']):
            return [
                "Where can I find detailed information about specific programs?",
                "How do I navigate to the academic programs section?",
                "Where are the degree requirements and curriculum details?",
                "How do I find contact information for the department?"
            ]
        elif any(word in goal_lower for word in ['search', 'find', 'look']):
            return [
                "Where is the search interface located on the page?",
                "What search options or filters are available?",
                "How do I refine or modify search results?",
                "What information can I search for in this application?"
            ]
        elif any(word in goal_lower for word in ['create', 'add', 'new', 'make']):
            return [
                "How do I access the creation interface or button?",
                "What information is required to create a new item?",
                "Where can I find the form or modal for creation?",
                "What happens after successfully creating an item?"
            ]
        elif any(word in goal_lower for word in ['edit', 'update', 'modify', 'change']):
            return [
                "How do I access editing functionality for existing items?",
                "What fields can be modified during editing?",
                "Where are the edit buttons or options located?",
                "How do I save changes after editing?"
            ]
        elif any(word in goal_lower for word in ['delete', 'remove', 'archive']):
            return [
                "How do I access deletion options?",
                "What confirmation steps are required for deletion?",
                "Where are delete buttons typically located?",
                "What happens to deleted items?"
            ]
        elif any(word in goal_lower for word in ['filter', 'sort', 'organize']):
            return [
                "Where is the filtering interface located?",
                "What filtering criteria are available?",
                "How do I apply and clear filters?",
                "Can filter settings be saved or customized?"
            ]
        else:
            # Generic exploration questions
            return [
                "How do I access the main functionality for this task?",
                "What are the primary navigation paths in the application?",
                "Where are the key interactive elements located?",
                "What is the typical workflow from start to finish?"
            ]
    
    def update_conversation_context(self, last_question, last_response):
        """Update conversation context for more coherent follow-ups"""
        # Ensure we have proper data to store
        if last_question and last_response:
            self.conversation_history.append({
                "last_question": last_question,
                "last_response": last_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only recent history to avoid context overflow
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
            
            print(f"💾 Updated conversation history with: Q: {last_question[:50]}...")
        else:
            print("⚠️  Skipping conversation update - missing question or response")
            

class IntelligentAgentA:
    """Enhanced Agent A using Hugging Face DialoGPT for question generation"""
    def __init__(self):
        self.dialogpt_agent = HuggingFaceDialoGPTAgent()
        self.current_goal = None
        self.exploration_context = {}
    
    
    def analyze_goal(self, user_input):
        """Analyze user goal and generate exploration strategy using DialoGPT"""
        self.current_goal = user_input
        self.exploration_context = {
            "goal": user_input,
            "started_at": datetime.now().isoformat(),
            "states_captured": 0,
            "elements_found": 0,
            "pages_visited": 0
        }
        
        # Generate initial questions using DialoGPT
        questions = self.dialogpt_agent.generate_questions(user_input)

        return questions
    


    def get_followup_questions(self, previous_question, exploration_results):
        """Generate follow-up questions based on exploration results"""
        if not exploration_results or not exploration_results.get('states'):
            print("❌ No exploration results for follow-up questions")
            return self.dialogpt_agent.generate_questions(self.current_goal)
        
        # Get the most recent state for context
        latest_state = exploration_results['states'][-1]
        context_summary = self._summarize_exploration_context(latest_state)
        
        print(f"📝 Context for follow-up: {context_summary}")
        
        # Update conversation context with meaningful data
        if previous_question and context_summary:
            self.dialogpt_agent.update_conversation_context(
                previous_question, 
                context_summary
            )
        else:
            print("⚠️  Cannot update conversation context - missing data")
        
        # Generate context-aware follow-up questions
        followup_questions = self.dialogpt_agent.generate_questions(
            self.current_goal, 
            context_summary,  # Pass the actual context
            max_questions=1
        )
        
        # Update exploration context
        self.exploration_context["states_captured"] = len(exploration_results['states'])
        self.exploration_context["pages_visited"] = len(set(
            state.get('url', '') for state in exploration_results['states']
        ))
        
        print(f"🔄 Generated {len(followup_questions)} follow-up questions")
        return followup_questions

    


    def _summarize_exploration_context(self, state):
        """Summarize the current exploration context for better follow-up questions"""
        if not state:
            return "No UI state captured yet."
        
        elements_count = len(state.get('interactive_elements', []))
        has_forms = state.get('form_count', 0) > 0
        has_modals = state.get('has_modals', False)
        current_page = state.get('title', 'Unknown page')
        current_url = state.get('url', '')
        
        # Count different types of elements
        buttons = [e for e in state.get('interactive_elements', []) if e.get('tag') == 'button']
        inputs = [e for e in state.get('interactive_elements', []) if e.get('tag') == 'input']
        links = [e for e in state.get('interactive_elements', []) if e.get('tag') == 'a']
        
        summary = f"Currently on page: '{current_page}'. "
        summary += f"Found {elements_count} interactive elements "
        summary += f"({len(buttons)} buttons, {len(inputs)} inputs, {len(links)} links)."
        
        if has_forms:
            summary += " There are forms available."
        if has_modals:
            summary += " Modal dialogs are present."
        
        # Add notable interactive elements with more context
        notable_elements = []
        for element in state.get('interactive_elements', [])[:10]:
            element_text = element.get('text', '').strip()
            element_type = element.get('type', '')
            element_placeholder = element.get('placeholder', '')
            element_tag = element.get('tag', '')
            
            if element_text and len(element_text) > 2 and len(element_text) < 50:
                notable_elements.append(f"'{element_text}'")
            elif element_placeholder:
                notable_elements.append(f"{element_tag}[{element_placeholder}]")
            elif element_type:
                notable_elements.append(f"{element_tag}[{element_type}]")
            elif element_tag in ['button', 'a']:
                notable_elements.append(f"{element_tag}")
        
        if notable_elements:
            summary += f" Notable elements: {', '.join(notable_elements[:6])}"
        
        # Add page context
        if 'admission' in current_url.lower() or 'admission' in current_page.lower():
            summary += " We are in an admissions-related section."
        elif 'graduate' in current_url.lower() or 'graduate' in current_page.lower():
            summary += " We are in a graduate programs section."
        elif 'program' in current_url.lower() or 'program' in current_page.lower():
            summary += " We are in a programs section."
        
        return summary


            
class BrowserAgentB:
    """Agent B with actual browser automation for screenshot capture"""
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.captured_states = []
        self.google_handler = None
    
    async def setup(self):
        """Initialize browser automation with bot avoidance"""
        self.playwright = await async_playwright().start()
        
        # Launch browser with bot-avoidance techniques
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-features=VizDisplayCompositor',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-web-security',
                '--disable-features=TranslateUI',
                '--disable-ipc-flooding-protection'
            ]
        )
        
        # Create context with more natural user agent and viewport
        context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            permissions=['notifications']
        )
        
        # Remove webdriver property
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
        self.page = await context.new_page()
        
        # Additional bot avoidance
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        """)
        
        # Initialize Google sign-in handler
        self.google_handler = GoogleSignInHandler(self.page)

    async def handle_possible_signin(self, state_data):
        """Check if sign-in is needed and handle it appropriately"""
        try:
            # Check for common sign-in indicators
            signin_indicators = [
                'sign in', 'login', 'log in', 'signin', 'account'
            ]
            
            page_title = state_data.get('title', '').lower()
            current_url = state_data.get('url', '').lower()
            
            # Check if we're on a sign-in page
            is_signin_page = any(indicator in page_title for indicator in signin_indicators) or \
                           any(indicator in current_url for indicator in signin_indicators)
            
            if is_signin_page:
                print("🔍 Detected sign-in page, checking for Google sign-in...")
                
                # Check for Google sign-in option
                if await self.google_handler.detect_google_signin():
                    print("🔄 Found Google sign-in option...")
                    
                    # Instead of automated login, wait for manual completion
                    success = await self.google_handler.wait_for_manual_google_signin()
                    
                    if success:
                        # Capture state after successful login
                        new_state = await self.capture_ui_state("After manual Google sign-in", "post_login")
                        print("✅ Manual Google sign-in completed successfully")
                        return new_state
                    else:
                        print("❌ Manual Google sign-in failed or timed out")
                        # Continue with current state even if sign-in fails
                        return state_data
            
            return state_data
            
        except Exception as e:
            print(f"⚠️ Error handling sign-in: {e}")
            return state_data
    
    async def capture_ui_state(self, description, state_type="intermediate"):
        """Capture comprehensive UI state with screenshot"""
        await self.page.wait_for_timeout(2000)  # Wait before capturing
        timestamp = datetime.now().isoformat()
        
        # Take screenshot
        screenshot_bytes = await self.page.screenshot(full_page=True)
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        screenshot_data = f"data:image/png;base64,{screenshot_b64}"
        
        # Get current URL and page info
        current_url = self.page.url
        page_title = await self.page.title()
        
        # Extract DOM state
        dom_state = await self.page.evaluate("""
            () => {
                const elements = [];
                // Capture interactive elements
                const selectors = ['button', 'a', 'input', 'select', 'textarea', '[role="button"]', '[onclick]'];
                
                selectors.forEach(selector => {
                    const foundElements = document.querySelectorAll(selector);
                    for (let i = 0; i < Math.min(foundElements.length, 50); i++) {
                        const el = foundElements[i];
                        if (el.offsetWidth > 0 && el.offsetHeight > 0) {
                            const rect = el.getBoundingClientRect();
                            elements.push({
                                tag: el.tagName.toLowerCase(),
                                text: el.textContent ? el.textContent.slice(0, 100).trim() : '',
                                type: el.type || '',
                                placeholder: el.placeholder || '',
                                id: el.id || '',
                                classes: el.className || '',
                                visible: el.checkVisibility ? el.checkVisibility() : true,
                                position: { 
                                    x: Math.round(rect.x), 
                                    y: Math.round(rect.y), 
                                    width: Math.round(rect.width), 
                                    height: Math.round(rect.height) 
                                }
                            });
                        }
                    }
                });
                
                return {
                    elements: elements,
                    url: window.location.href,
                    hasModals: document.querySelectorAll('[role="dialog"], .modal, .popup').length > 0,
                    formCount: document.querySelectorAll('form').length
                };
            }
        """)
        
        state_data = {
            "timestamp": timestamp,
            "description": description,
            "type": state_type,
            "url": current_url,
            "title": page_title,
            "screenshot": screenshot_data,
            "interactive_elements": dom_state["elements"],
            "has_modals": dom_state["hasModals"],
            "form_count": dom_state["formCount"],
            "viewport": {"width": 1280, "height": 720}
        }

        # Check if we need to handle sign-in
        state_data = await self.handle_possible_signin(state_data)
        
        self.captured_states.append(state_data)
        return state_data
    
    async def navigate_to(self, url, description):
        """Navigate to URL and capture initial state"""
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=45000)
            await asyncio.sleep(3)
            # Add random delays to appear more human
            await asyncio.sleep(1 + random.random() * 2)
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Wait for random time to appear human
            await asyncio.sleep(2 + random.random() * 3)
            
            await self.page.wait_for_timeout(2000)
            return await self.capture_ui_state(description, "initial")
        except Exception as e:
            print(f"Navigation failed: {e}")
            try:
                return await self.capture_ui_state(f"Failed navigation: {description}", "error")
            except:
                return None
    
    async def click_element(self, selector, description):
        """Click element and capture resulting state"""
        try:
            # Add human-like delay
            await asyncio.sleep(0.5 + random.random() * 1)
            
            await self.page.click(selector)
            # Wait for human-like time
            await asyncio.sleep(2 + random.random() * 2)
            await self.page.wait_for_timeout(4000)
            return await self.capture_ui_state(description, "interaction")
        except Exception as e:
            print(f"Click failed: {e}")
            return await self.capture_ui_state(f"Failed: {description}", "error")
    
    async def click_element_by_text(self, text, description):
        """Click element by text content"""
        try:
            await self.page.click(f"text={text}")
            await self.page.wait_for_timeout(4000)
            return await self.capture_ui_state(description, "interaction")
        except Exception as e:
            print(f"Click by text failed: {e}")
            return await self.capture_ui_state(f"Failed: {description}", "error")
    
    async def explore_autonomously(self, goal_description):
        """Autonomously explore the application based on goal"""
        exploration_states = []
        
        strategies = [
            ("button", ['Create', 'Add', 'New', 'Filter', 'Search']),
            ("navigation", ['Projects', 'Tasks', 'Issues', 'Database']),
        ]
        
        for element_type, keywords in strategies:
            for keyword in keywords:
                try:
                    if element_type == "button":
                        await self.page.click(f"button:has-text('{keyword}')", timeout=4000)
                        state = await self.capture_ui_state(
                            f"Autonomous: clicked {keyword} button", 
                            "autonomous"
                        )
                        exploration_states.append(state)
                        await self.page.wait_for_timeout(2000)
                        break
                except:
                    continue
        
        return exploration_states
    
    async def close(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

class EnhancedMultiAgentSystem:
    """Enhanced system with Hugging Face integration"""
    def __init__(self):
        self.agent_a = IntelligentAgentA()
        self.agent_b = BrowserAgentB()
        self.conversation_history = []

    def set_huggingface_token(self, token):
        """Set Hugging Face API token"""
        #self.agent_a.set_api_token(token)
    
    async def initialize(self):
        """Initialize the system"""
        await self.agent_b.setup()
    
    async def execute_workflow(self, user_goal, app_url, max_steps=6):
        """Execute complete workflow with AI-generated questions and follow-ups"""
        print(f"\n🎯 Starting workflow: {user_goal}")
        print(f"🌐 Application: {app_url}")
        
        # Initial navigation
        print("🔄 Navigating to application...")
        initial_state = await self.agent_b.navigate_to(app_url, f"Initial load: {app_url}")
        
        if initial_state is None:
            print("❌ Failed to initialize - cannot continue")
            return []
            
        self.conversation_history.append({
            "step": 0,
            "action": "navigate",
            "description": f"Initial navigation to {app_url}",
            "state": initial_state
        })
        
        #print(f"📄 Loaded: {initial_state['title']}")

        # Check if we're on a sign-in page and handle it
        current_url = initial_state.get('url', '').lower()
        if any(indicator in current_url for indicator in ['login', 'signin', 'auth', 'accounts.google.com']):
            print("🔐 Application requires authentication")
            print("   The system will pause for manual sign-in if needed...")
            
            # Let the sign-in handler in capture_ui_state handle this
            # We don't need to do anything extra here
            
        
        # Get AI-generated exploration strategy from Agent A
        initial_questions = self.agent_a.analyze_goal(user_goal)
        print(f"🔍 AI Strategy: {len(initial_questions)} initial questions")
        for i, question in enumerate(initial_questions, 1):
            print(f"   {i}. {question}")
        
        # Execute initial questions
        executed_questions = []
        current_questions = initial_questions[:max_steps//2]  # Use half for initial questions

        
        for step, question in enumerate(current_questions, 1):
            #print(f"\n🔄 Question {step}/{max_steps}: {question}")
            print(f"\n🔄 Question {step}: {question}")
            
            # Convert question to action
            action_result = await self._execute_question(question, user_goal)
            
            if action_result:
                self.conversation_history.append({
                    "step": step,
                    "question": question,
                    "action": "exploration",
                    "state": action_result
                })
                executed_questions.append(question)
                print(f"✅ Captured state with {len(action_result['interactive_elements'])} elements")
                
                # Generate follow-up questions after each exploration
                if step >= 1:  # Start generating follow-ups after first exploration
                    exploration_results = {
                        "states": [item["state"] for item in self.conversation_history],
                        "current_state": action_result
                    }
                    
                    followup_questions = self.agent_a.get_followup_questions(
                        question, 
                        exploration_results
                    )
                    
                    if followup_questions and len(executed_questions) < max_steps:
                        # Add follow-up questions to current queue
                        current_questions.extend(followup_questions)
                        print(f"🔄 Generated {len(followup_questions)} follow-up questions")
                        for i, fq in enumerate(followup_questions, 1):
                            print(f"   ➕ {fq}")
            else:
##                print("❌ Failed to capture state")
                print("⏭️ No state captured for this question - continuing to next")
                executed_questions.append(question)  # Still count as executed
        
        # Execute remaining follow-up questions
        remaining_steps = max_steps - len(executed_questions)
        if remaining_steps > 0 and len(current_questions) > len(executed_questions):
            followup_queue = current_questions[len(executed_questions):]
            followup_queue = followup_queue[:remaining_steps]  # Take only remaining steps
            
            for step, question in enumerate(followup_queue, len(executed_questions) + 1):
                print(f"\n🔄 Step {step}/{max_steps} (Follow-up): {question}")
                
                action_result = await self._execute_question(question, user_goal)
                
                if action_result:
                    self.conversation_history.append({
                        "step": step,
                        "question": question,
                        "action": "followup_exploration",
                        "state": action_result
                    })
                    executed_questions.append(question)
                    print(f"✅ Captured state with {len(action_result['interactive_elements'])} elements")
        
        # Autonomous exploration for non-URL states
        if self.conversation_history:
            print("\n🔍 Starting autonomous exploration...")
            exploration_states = await self.agent_b.explore_autonomously(user_goal)
            for i, state in enumerate(exploration_states):
                self.conversation_history.append({
                    "step": len(executed_questions) + i + 1,
                    "action": "autonomous_exploration", 
                    "description": f"Autonomous exploration {i+1}",
                    "state": state
                })
        
        # Save complete workflow
        if self.conversation_history:
            workflow_dir = await self._save_workflow(user_goal, app_url)
            print(f"\n📊 Workflow completed: {len(self.conversation_history)} states captured")
            print(f"💾 Saved to: {workflow_dir}")
            
            # Show AI learning summary
            self._show_ai_learning_summary(user_goal)
            
            # Show question statistics
            total_questions = len([item for item in self.conversation_history if item.get("question")])
            followup_questions = len([item for item in self.conversation_history if item.get("action") == "followup_exploration"])
            print(f"❓ Questions: {total_questions} total ({followup_questions} follow-ups)")
        else:
            print("❌ No states captured - workflow failed")
        
        return self.conversation_history
    

    async def _execute_question(self, question, goal):
        """Execute action based on AI-generated question with generic pattern matching"""
        question_lower = question.lower()

        # Store current URL before any action
        current_url_before = self.agent_b.page.url
        
        # Generic pattern matching for different question types
        if any(word in question_lower for word in ['where', 'navigation', 'access', 'located', 'find', 'locate']):
            return await self._explore_navigation(question, goal)
        elif any(word in question_lower for word in ['button', 'click', 'interact', 'press', 'tap']):
            return await self._explore_interactions(question, goal)
        elif any(word in question_lower for word in ['form', 'field', 'input', 'required', 'enter', 'type', 'fill']):
            return await self._explore_forms(question, goal)
        elif any(word in question_lower for word in ['search', 'query', 'lookup', 'find', 'look']):
            return await self._explore_search(question, goal)
        elif any(word in question_lower for word in ['menu', 'dropdown', 'select', 'choose']):
            return await self._explore_navigation(question, goal)  # Treat as navigation
        else:
            # For ambiguous questions, try navigation first, then interactions
            result = await self._explore_navigation(question, goal)
            if result and len(result.get('interactive_elements', [])) > 0:
                return result
            return await self._explore_interactions(question, goal)
        
        # Only return result if page actually changed
        if result and self.agent_b.page.url != current_url_before:
            print(f"✅ Meaningful change detected - proceeding with capture")
            return result
        else:
            print("⏭️ Skipping capture - no page change detected")
            return None
    


    async def _explore_navigation(self, question, goal):
        """Explore navigation elements with intelligent generic matching"""
        try:
            # Comprehensive navigation selectors for all websites
            nav_selectors = [
                'nav a', '[role="navigation"] a', '.navbar a', '.menu a', 'header a',
                '.main-nav a', '.primary-nav a', '.site-nav a', '#main-nav a',
                '.navigation a', '.nav-menu a', '.menu-item a', '.nav-link',
                '.header a', '.top-nav a', '.main-menu a', '.primary-menu a',
                '[class*="nav"] a', '[class*="menu"] a', '[class*="header"] a'
            ]
            
            # Extract meaningful keywords from goal and question
            goal_keywords = self._extract_keywords(goal)
            question_keywords = self._extract_keywords(question)
            all_keywords = list(set(goal_keywords + question_keywords))
            
            print(f"🔍 Looking for navigation matching keywords: {all_keywords}")

            # Store current URL before navigation attempts
            current_url_before = self.agent_b.page.url
            
            # Try each navigation selector
            for selector in nav_selectors:
                try:
                    elements = await self.agent_b.page.query_selector_all(selector)
                    print(f"📋 Found {len(elements)} elements with selector: {selector}")
                    
                    for element in elements[:10]:  # Check first 10 elements per selector
                        try:
                            text = await element.text_content()
                            if text and text.strip():
                                text_clean = text.strip()
                                text_lower = text_clean.lower()
                                
                                # Check for keyword matches with flexible matching
                                matches_keyword = self._matches_any_keyword(text_lower, all_keywords)
                                
                                # Check for common action words that might be relevant
                                common_actions = ['learn more', 'get started', 'explore', 'view', 'see', 'discover']
                                matches_action = any(action in text_lower for action in common_actions)
                                
                                # Check if text seems like a meaningful navigation item
                                is_meaningful_nav = (
                                    len(text_clean) > 2 and 
                                    len(text_clean) < 50 and
                                    not text_lower in ['home', 'login', 'sign in', 'contact'] and
                                    any(char.isalpha() for char in text_clean)
                                )
                                
                                if (matches_keyword or matches_action) and is_meaningful_nav:
                                    print(f"🎯 Clicking navigation: '{text_clean}'")
                                    
                                    # Scroll element into view and click
                                    await element.scroll_into_view_if_needed()
                                    await element.click()
                                    await self.agent_b.page.wait_for_timeout(3000)

                                    # Only capture if URL actually changed
                                    if self.agent_b.page.url != current_url_before:
                                        print(f"✅ Page changed from {current_url_before} to {self.agent_b.page.url}")
                                        # Capture the resulting state
                                        return await self.agent_b.capture_ui_state(
                                            f"Navigation: clicked '{text_clean}'", 
                                            "navigation"
                                        )
                                    else:
##                                        print("⏭️ No page change detected after click, trying next element")
##                                        continue
                                        break
                                
                                    
##                                    # Capture the resulting state
##                                    return await self.agent_b.capture_ui_state(
##                                        f"Navigation: clicked '{text_clean}'", 
##                                        "navigation"
##                                    )
                        except Exception as e:
                            print(f"⚠️ Error clicking element: {e}")
                            continue
                except Exception as e:
                    print(f"⚠️ Error with selector {selector}: {e}")
                    continue
            
            # Fallback: try to find any meaningful navigation
            print("🔍 Trying fallback navigation discovery...")
            fallback_result = await self._explore_fallback_navigation(goal, all_keywords)
##            if fallback_result:
##                return fallback_result

            # Check if fallback actually changed the page
            if fallback_result and self.agent_b.page.url != current_url_before:
                return fallback_result
            else:
                print("⏭️ Fallback navigation didn't change the page")
                return None
        
                
        except Exception as e:
            print(f"❌ Navigation exploration error: {e}")
        
##        # Final fallback - capture current state
##        return await self.agent_b.capture_ui_state(f"Navigation exploration: {question}", "navigation_scan")

        # Final fallback - only capture if page changed
        if self.agent_b.page.url != current_url_before:
            return await self.agent_b.capture_ui_state(f"Navigation exploration: {question}", "navigation_scan")
        else:
            print("⏭️ No page change detected in navigation exploration")
            return None
        


    async def _explore_interactions(self, question, goal):
        """Explore interactive elements with generic pattern matching"""
        try:
            # Extract keywords from goal and question
            goal_keywords = self._extract_keywords(goal)
            question_keywords = self._extract_keywords(question)
            all_keywords = list(set(goal_keywords + question_keywords))
            
            # Common interactive element texts across all websites
            common_interactions = [
                'search', 'find', 'explore', 'discover', 'learn more', 'get started',
                'view', 'see', 'browse', 'filter', 'sort', 'create', 'add', 'new',
                'edit', 'update', 'manage', 'settings', 'options', 'preferences'
            ]
            
            # Combine with extracted keywords
            all_button_texts = list(set(common_interactions + all_keywords))
            
            print(f"🔍 Looking for interactions with: {all_button_texts}")
            
            # Try different element types
            element_types = ['button', 'a', '[role="button"]', '[onclick]']
            
            for element_type in element_types:
                for button_text in all_button_texts:
                    if len(button_text) > 2:
                        try:
                            # Try to find elements with this text
                            selector = f"{element_type}:has-text('{button_text}')"
                            elements = await self.agent_b.page.query_selector_all(selector)
                            
                            if not elements:
                                # Try case-insensitive match
                                selector = f"{element_type}:has-text(/{button_text}/i)"
                                elements = await self.agent_b.page.query_selector_all(selector)
                            
                            for element in elements[:3]:  # Try first 3 matches
                                try:
                                    text = await element.text_content()
                                    if text and text.strip():
                                        print(f"🎯 Clicking {element_type}: '{text.strip()}'")
                                        await element.scroll_into_view_if_needed()
                                        await element.click()


                                        # Wait and check if URL changed
                                        await self.agent_b.page.wait_for_timeout(2000)
                                        
                                        # Only capture if URL actually changed
                                        if self.agent_b.page.url != current_url_before:
                                            return await self.agent_b.capture_ui_state(
                                                f"Interaction: clicked '{text.strip()}'", 
                                                "interaction"
                                            )
                                        else:
                                            print("⏭️ No page change detected after interaction")
                                            continue
                                        
##                                        return await self.agent_b.capture_ui_state(
##                                            f"Interaction: clicked '{text.strip()}'", 
##                                            "interaction"
##                                        )
                                except:
                                    continue
                        except:
                            continue
                        
        except Exception as e:
            print(f"❌ Interaction exploration error: {e}")
        
##        return await self.agent_b.capture_ui_state(f"Interaction exploration: {question}", "interaction_scan")

        # Final fallback - only capture if page changed
        if self.agent_b.page.url != current_url_before:
            return await self.agent_b.capture_ui_state(f"Interaction exploration: {question}", "interaction_scan")
        else:
            print("⏭️ No page change detected in interaction exploration")
            return None
    
    async def _explore_forms(self, question, goal):
        """Explore form elements"""
        return await self.agent_b.capture_ui_state(f"Form analysis: {question}", "form_analysis")
    
    async def _explore_search(self, question, goal):
        """Explore search functionality"""
        try:
            # Look for search inputs
            search_selectors = ['input[type="search"]', 'input[placeholder*="search"]', '[aria-label*="search"]']
            for selector in search_selectors:
                try:
                    element = await self.agent_b.page.query_selector(selector)
                    if element:
                        await element.click()
                        await self.agent_b.page.wait_for_timeout(1000)
                        return await self.agent_b.capture_ui_state(
                            f"Search interface: found search input", 
                            "search_analysis"
                        )
                except:
                    continue
        except Exception as e:
            print(f"Search exploration error: {e}")
        
        return await self.agent_b.capture_ui_state(f"Search exploration: {question}", "search_scan")
    
    async def _is_goal_achieved(self, goal, state):
        """Check if goal has been achieved"""
        if not state:
            return False
        
        current_url = state.get('url', '').lower()
        success_indicators = ['success', 'created', 'added', 'complete', 'thankyou']
        
        if any(indicator in current_url for indicator in success_indicators):
            return True
        
        page_title = state.get('title', '').lower()
        if any(indicator in page_title for indicator in success_indicators):
            return True
        
        elements = state.get('interactive_elements', [])
        for element in elements:
            element_text = element.get('text', '').lower()
            if any(indicator in element_text for indicator in success_indicators):
                return True
        
        return False
    
    def _show_ai_learning_summary(self, user_goal):
        """Show what the AI learned during exploration"""
        if hasattr(self.agent_a.dialogpt_agent, 'conversation_history'):
            history = self.agent_a.dialogpt_agent.conversation_history
            if history:
                print(f"\n🧠 AI Learning Summary for '{user_goal}':")
                print(f"   - Generated {len(history)} question sets")
                print(f"   - Adapted questions based on UI discoveries")
                print(f"   - Maintained conversation context across {len(self.conversation_history)} states")
    
    async def _save_workflow(self, user_goal, app_url):
        """Save complete workflow with screenshots"""
        goal_slug = re.sub(r'[^a-zA-Z0-9]', '_', user_goal.lower())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_dir = f"workflows/{goal_slug}_{timestamp}"
        os.makedirs(workflow_dir, exist_ok=True)
        
        workflow_data = {
            "goal": user_goal,
            "app_url": app_url,
            "captured_at": datetime.now().isoformat(),
            "total_states": len(self.conversation_history),
            "ai_generated_questions": [
                item.get("question") for item in self.conversation_history if item.get("question")
            ],
            "states": []
        }
        
        # Save individual screenshots and state data
        for i, item in enumerate(self.conversation_history):
            state = item["state"]
            
            # Save screenshot as file
            screenshot_data = state["screenshot"]
            if screenshot_data.startswith("data:image/png;base64,"):
                screenshot_b64 = screenshot_data.split(",")[1]
                screenshot_bytes = base64.b64decode(screenshot_b64)
                with open(f"{workflow_dir}/state_{i:02d}.png", "wb") as f:
                    f.write(screenshot_bytes)
            
            # Store metadata in JSON
            workflow_data["states"].append({
                "step": item["step"],
                "action": item.get("action", "unknown"),
                "description": item.get("description", ""),
                "question": item.get("question", ""),
                "timestamp": state["timestamp"],
                "url": state["url"],
                "title": state["title"],
                "screenshot_file": f"state_{i:02d}.png",
                "interactive_elements_count": len(state["interactive_elements"]),
                "has_modals": state["has_modals"],
                "sample_elements": state["interactive_elements"][:5]
            })
        
        # Save workflow data
        with open(f"{workflow_dir}/workflow.json", "w", encoding='utf-8') as f:
            json.dump(workflow_data, f, indent=2, ensure_ascii=False)
        
        return workflow_dir
    
    async def close(self):
        """Clean up resources"""
        await self.agent_b.close()

    def _extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        if not text:
            return []
        
        # Remove common stop words and extract meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Clean and split text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and return unique words
        keywords = [word for word in words if word not in stop_words]
        
        return list(set(keywords))  # Return unique keywords

    def _matches_any_keyword(self, text, keywords):
        """Check if text matches any keyword with flexible matching"""
        if not text or not keywords:
            return False
        
        text_lower = text.lower()
        
        for keyword in keywords:
            # Exact match
            if keyword in text_lower:
                return True
            # Partial match (keyword contained in text)
            if any(keyword in word for word in text_lower.split()):
                return True
            # Stemmed matching for common variations
            if self._stemmed_match(keyword, text_lower):
                return True
        
        return False

    def _stemmed_match(self, keyword, text):
        """Simple stemmed matching for common word variations"""
        stems = {
            'apply': ['apply', 'application', 'applying'],
            'create': ['create', 'creating', 'creation'],
            'search': ['search', 'searching', 'searches'],
            'find': ['find', 'finding', 'found'],
            'program': ['program', 'programs', 'programming'],
            'project': ['project', 'projects'],
            'task': ['task', 'tasks'],
            'manage': ['manage', 'managing', 'management'],
            'setting': ['setting', 'settings'],
            'profile': ['profile', 'profiles'],
            'account': ['account', 'accounts'],
            'user': ['user', 'users'],
            'admin': ['admin', 'administrator'],
            'help': ['help', 'support'],
            'guide': ['guide', 'guides', 'guidance']
        }
        
        for base, variations in stems.items():
            if keyword == base:
                return any(variation in text for variation in variations)
            if keyword in variations:
                return any(variation in text for variation in variations)
        
        return False

    async def _explore_fallback_navigation(self, goal, keywords):
        """Fallback navigation discovery using multiple strategies"""
        try:
            # Strategy 1: Look for any links with relevant text
            all_links = await self.agent_b.page.query_selector_all('a')
            
            for link in all_links[:20]:  # Check first 20 links
                try:
                    text = await link.text_content()
                    if text and text.strip():
                        text_clean = text.strip()
                        text_lower = text_clean.lower()
                        
                        # Check if link text seems meaningful and relevant
                        is_relevant = (
                            len(text_clean) > 3 and 
                            len(text_clean) < 100 and
                            any(keyword in text_lower for keyword in keywords) and
                            not text_lower in ['skip to content', 'privacy policy', 'terms of service']
                        )
                        
                        if is_relevant:
                            print(f"🎯 Clicking fallback link: '{text_clean}'")
                            await link.scroll_into_view_if_needed()
                            await link.click()
                            await self.agent_b.page.wait_for_timeout(3000)
                            return await self.agent_b.capture_ui_state(
                                f"Fallback navigation: clicked '{text_clean}'", 
                                "navigation"
                            )
                except:
                    continue
            
            # Strategy 2: Look for buttons with relevant text
            all_buttons = await self.agent_b.page.query_selector_all('button')
            
            for button in all_buttons[:15]:  # Check first 15 buttons
                try:
                    text = await button.text_content()
                    if text and text.strip():
                        text_clean = text.strip()
                        text_lower = text_clean.lower()
                        
                        # Check if button text seems like navigation
                        is_navigation_button = (
                            any(keyword in text_lower for keyword in keywords) and
                            len(text_clean) < 30 and
                            not text_lower in ['submit', 'cancel', 'close']
                        )
                        
                        if is_navigation_button:
                            print(f"🎯 Clicking fallback button: '{text_clean}'")
                            await button.scroll_into_view_if_needed()
                            await button.click()
                            await self.agent_b.page.wait_for_timeout(3000)
                            return await self.agent_b.capture_ui_state(
                                f"Fallback button: clicked '{text_clean}'", 
                                "navigation"
                            )
                except:
                    continue
                    
        except Exception as e:
            print(f"❌ Fallback navigation error: {e}")
        
        return None

# Interactive function with Hugging Face integration
async def interactive_tasks_with_ai():
    """Interactive function with AI-powered question generation"""
    system = EnhancedMultiAgentSystem()
    
    try:
        print("🤖 AI-POWERED MULTI-AGENT UI CAPTURE SYSTEM")
        print("=" * 60)
        print("This system uses Hugging Face's DialoGPT-small to generate")
        print("intelligent questions for exploring any web application.")
        print("This system supports Google sign-in with manual completion")
        print("When Google sign-in is detected, you'll be prompted to")
        print("complete the authentication manually in the browser.")
        print("=" * 60)
        

        
        example_tasks = [
            {"goal": "Search articles in Wikipedia", "url": "https://en.wikipedia.org/wiki/Main_Page"},
            {"goal": "Create project in Linear", "url": "https://linear.app"},
            {"goal": "Filter database in Notion", "url": "https://www.notion.so"},
            {"goal": "Create task in Asana", "url": "https://app.asana.com"},
            {"goal": "Search products on Amazon", "url": "https://www.amazon.com"},
            {"goal": "Find repositories on GitHub", "url": "https://github.com"},
        ]
        
        while True:
            print(f"\n📋 EXAMPLE TASKS:")
            for i, task in enumerate(example_tasks, 1):
                print(f"  {i}. {task['goal']}")
                print(f"     URL: {task['url']}")
            
            print(f"\n🎯 ENTER YOUR OWN TASK (or 'quit' to exit):")
            user_goal = input("What do you want to learn how to do? ").strip()
            
            if user_goal.lower() == 'quit':
                break
                
            app_url = input("Enter the application URL: ").strip()
            
            if not user_goal or not app_url:
                print("❌ Please provide both a goal and URL")
                continue
            
            try:
                print(f"\n🚀 Starting AI-powered exploration...")
                print(f"Goal: {user_goal}")
                print(f"URL: {app_url}")
                print("🔐 Google sign-in will be handled manually if detected")
                print("   A browser window will open - please complete any")
                print("   required sign-in processes when prompted.")
                
                # Initialize and execute
                await system.initialize()
                workflow = await system.execute_workflow(user_goal, app_url, max_steps=4)
                
                if workflow:
                    print(f"\n🎉 AI EXPLORATION COMPLETE!")
                    print(f"✅ Captured {len(workflow)} UI states")
                    print(f"📸 Screenshots and AI questions saved in workflows/ directory")
                    print(f"🧠 The AI generated context-aware questions based on discoveries")
                    
                else:
                    print(f"❌ No states captured.")
                    
            except Exception as e:
                print(f"❌ Error during exploration: {e}")
            finally:
                # Reset for next task
                system.conversation_history = []
                await system.agent_b.close()
                system.agent_b = BrowserAgentB()
            
            print(f"\n{'='*60}")
            print("Ready for another task!")
            print("=" * 60)
                
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await system.close()
        print("\n👋 Thank you for using the AI-Powered Multi-Agent System!")

# Test function
async def test_ai_system():
    """Test the AI-powered system"""
    system = EnhancedMultiAgentSystem()
    
    try:
        print("🚀 Testing AI-Powered Multi-Agent System...")
        await system.initialize()
        
        # Test with Wikipedia
        workflow = await system.execute_workflow(
            "Search articles in Wikipedia", 
            "https://en.wikipedia.org/wiki/Main_Page",
            max_steps=1
        )
        
        if workflow:
            print(f"✅ Successfully captured {len(workflow)} UI states with AI-generated questions")
        else:
            print("❌ No workflow captured")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await system.close()

if __name__ == "__main__":
    # Install required packages
    try:
        import playwright
        import requests
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "playwright", "requests"])
        subprocess.run(["playwright", "install", "chromium"])


    system = EnhancedMultiAgentSystem()
    file=open("hf_key.txt","r")
    hf_key=file.readline()
    system.set_huggingface_token(hf_key)
    print("🚀 AI-Powered Multi-Agent UI Capture System")
    print("Choose an option:")
    print("1. Run AI-powered test (Wikipedia)")
    print("2. Interactive mode with AI question generation")
    print("3. Quit")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\nRunning AI-powered test...")
        asyncio.run(test_ai_system())
    elif choice == "2":
        print("\nStarting interactive AI mode...")
        asyncio.run(interactive_tasks_with_ai())
    elif choice == "3":
        print("\nQuit...")
        exit()
    else:
        print("Invalid choice. Running AI-powered test...")
        asyncio.run(test_ai_system())
