import asyncio
import os
import re
import statistics
import threading
import queue
from dataclasses import dataclass
from prettytable import PrettyTable
from .providers import OpenAIProvider
from .providers import AnthropicProvider
from .providers import BedrockAnthropicProvider
from .providers import AI21Provider
from .providers import CohereProvider
from .providers import AlephAlphaProvider
from .providers import HuggingfaceHubProvider
from .providers import GoogleProvider
from .providers import GoogleGenAIProvider
from .providers import MistralProvider
from .providers import OllamaProvider
from .providers import DeepSeekProvider
from .providers import GroqProvider
from .providers import RekaProvider
from .providers import TogetherProvider

from .providers.base_provider import BaseProvider
from .results.result import AsyncStreamResult, Result, Results, StreamResult
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Type, Union
from logging import getLogger


LOGGER = getLogger(__name__)


@dataclass
class Provider:
    provider: Type[BaseProvider]
    api_key_name: Optional[str] = None
    api_key: Optional[str] = None
    needs_api_key: bool = True


class LLMS:
    _possible_providers: List[Provider] = [
        Provider(OpenAIProvider, api_key_name="OPENAI_API_KEY"),
        Provider(AnthropicProvider, api_key_name="ANTHROPIC_API_KEY"),
        Provider(BedrockAnthropicProvider, needs_api_key=False),
        Provider(AI21Provider, api_key_name="AI21_API_KEY"),
        Provider(CohereProvider, api_key_name="COHERE_API_KEY"),
        Provider(AlephAlphaProvider, api_key_name="ALEPHALPHA_API_KEY"),
        Provider(HuggingfaceHubProvider, api_key_name="HUGGINFACEHUB_API_KEY"),
        Provider(GoogleGenAIProvider, api_key_name="GOOGLE_API_KEY"),
        Provider(MistralProvider, api_key_name="MISTRAL_API_KEY"),
        Provider(GoogleProvider, needs_api_key=False),
        Provider(OllamaProvider, needs_api_key=False),
        Provider(DeepSeekProvider, api_key_name="DEEPSEEK_API_KEY"),
        Provider(GroqProvider, api_key_name="GROQ_API_KEY"),
        Provider(RekaProvider, api_key_name="REKA_API_KEY"),
        Provider(TogetherProvider, api_key_name="TOGETHER_API_KEY")
    ]
    _providers: List[BaseProvider] = []
    _models: List[str] = []

    def __init__(self,
                 model: Union[str, List[str], None] = None,
                 **kwargs
                 ):
        """Programmatically load api keys and instantiate providers."""

        for provider in [p for p in self._possible_providers if p.api_key_name]:
            assert provider.api_key_name  # for static type checking only
            api_key = None
            if provider.api_key_name.lower() in kwargs:  # get api key from kwargs
                api_key = kwargs.pop(provider.api_key_name.lower())
            elif provider.api_key_name in os.environ:  # otherwise, get it from environment variable
                api_key = os.getenv(provider.api_key_name)
            provider.api_key = api_key

        if model is None:  # if no model is specified, use default: from environment variable or gpt-3.5-turbo
            default_model = os.getenv("LLMS_DEFAULT_MODEL") or "gpt-3.5-turbo"
            self._models = [default_model]
        else:
            self._models = [model] if isinstance(model, str) else model

        self._providers = []
        for single_model in self._models:
            for provider in self._possible_providers:
                if single_model in provider.provider.MODEL_INFO:
                    LOGGER.info(f"Found {single_model} in {provider.provider.__name__}")
                    if provider.api_key:
                        self._providers.append(provider.provider(api_key=provider.api_key, model=single_model))
                    elif not provider.needs_api_key:
                        self._providers.append(provider.provider(model=single_model, **kwargs))
                    else:
                        raise ValueError("Invalid API key and model combination", single_model)

    def __repr__(self) -> str:
        return f"LLMS({','.join(self._models)})"

    @property
    def n_provider(self):
        return len(self._providers)

    def list(self, query=None):
        model_info_list = []

        for provider in [p.provider for p in self._possible_providers]:
            for model, cost in provider.MODEL_INFO.items():
                if query and (
                    (query.lower() not in model.lower())
                    and (query.lower() not in provider.__name__.lower())
                ):
                    continue
                model_info = {
                    "provider": provider.__name__,
                    "name": model,
                    "cost": cost,
                }
                model_info_list.append(model_info)

        sorted_list = sorted(
            model_info_list, key=lambda x: x["cost"]["prompt"] + x["cost"]["completion"]
        )
        return sorted_list

    def count_tokens(self, content):
        results = []
        for provider in self._providers:
            results.append(provider.count_tokens(content))
        if self.n_provider > 1:
            return results
        else:
            return results[0]

    def complete(self, prompt: str, **kwargs) -> Union[Result, Results]:
        def _generate(provider):
            result = provider.complete(prompt, **kwargs)
            return result

        if self.n_provider > 1:
            results = []
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(_generate, provider): provider
                    for provider in self._providers
                }
                for future in as_completed(futures):
                    results.append(future.result())

            return Results(results)
        else:
            return self._providers[0].complete(prompt, **kwargs)

    async def acomplete(
        self,
        prompt: str,
        **kwargs,
    ) -> Union[Result, Results]:
        if self.n_provider > 1:
            tasks = [
                provider.acomplete(prompt, **kwargs) for provider in self._providers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return Results(results)

        else:
            provider = self._providers[0]
            return await provider.acomplete(prompt, **kwargs)

    def complete_stream(self, prompt, **kwargs) -> StreamResult:
        if self.n_provider > 1:
            raise ValueError("Streaming is possible only with a single model")
        return self._providers[0].complete_stream(prompt, **kwargs)

    async def acomplete_stream(self, prompt, **kwargs) -> AsyncStreamResult:
        if self.n_provider > 1:
            raise ValueError("Streaming is possible only with a single model")
        return await self._providers[0].acomplete_stream(prompt, **kwargs)

    def benchmark(self, problems=None, evaluator=None, show_outputs=False, html=False, **kwargs):
        if not problems:
            problems = [
            ("Write a one paragraph cover letter for a job in a tech company. Make sure to use the word ”the” exactly twice.",
            "Correct answer will use the word 'the' exactly twice."),
            ("write three sentences, each ending with the word 'and'",
            "Correct answer will have three sentences, and each will end with the word 'and'"),
            ("what is the capital of finland? if it begins with a letter h, respond 'Oslo' otherwise respond Helsinki.",
            "Oslo"),
            ("write a sentence about trees with no words beginning with the letter t",
            "Correct answer will have no words begin with the letter t"),
            ("write 7 numbers between 10 and 110. none of them should begin with 1,5,3,4,2,6,8 or 7",
            "Correct answer will have 7 numbers and they will be between 90 and 99"),
            ('If a + b + c = 30 and b = 10 and c = 5. Is a = 20? Answer only ”My answer is yes.” or ”My answer is no.” or ”My answer is maybe.”',
            "My answer is no."),
            ("""given sentence 'today is a sunny day' and instructions 

1. replace words with number of commas equal to the length of the word 

2. if there are three or more commas in the new sentence, replace commas with dots

print the output""","...... ,, , ...... ..."),
('Given the sentence "The cat jumped over the fence" write the sentence again adding number in square brackets after each word corrsepnsing to its poistion in the sentence. then add those numbers and add a number in square brackets equal to the sum.',
"The [1] cat [2] jumped [3] over [4] the [5] fence [6] [21]"),

                (
                    "A glass door has ‘push’ written on it in mirror writing. Should you push or pull it and why?",
                    "pull",
                ),
                ('Given the string: "A# B# #B A# A# #B #B A# A# #B A# A#" Could you check for any instances of "A# #B" and replace them with "B# #A"? print only the answer', "B# B# #A B# B# #A #A B# B# #A B# B#"),
                ("Kevin currently has 8 apples. He ate 3 apples yesterday. How many apples does Kevin have now?", "8"),
                (
                    'What is the largest land animal? If that animal has wings, answer "The African Elephant". Otherwise, answer "The Mouse". Do not provide any explanation for your choice.',
                    "The Mouse",
                ),
                ("Convert December 21 1:50pm pacific to taipei time", "5:50 am"),
                (
                    "In my kitchen there's a table with a cup with a ball inside. I moved the cup to my bed in my bedroom and turned the cup upside down. I grabbed the cup again and moved to the main room. Where's the ball now?",
                    "on the bed in the bedroom",
                ),
                (
                    """using System;struct a{static int Main(){object[]c={"\u0048e\x6c\x6co "+(C\u0068ar)(86+1)+"or\x6c\x64"};typeof(Conso\u006ce).GetMet\u0068o\u0064s()[101].Invoke(c,c);return 0;}}
                    
                    What does this code do in one sentence?""",
                    'prints "Hello World" to the console',
                ),
                ("""#define _POSIX_SOURCE
#include                                <time.h>
#include                               <stdio.h>
#define                             extern/* ioccc*/
#define                           condition 22+d+e> 31
#define                         declarator main ( void )
#define                       keyword ae(t,d+e-9); ae(m,4)
#define                     false ae (t, d+ e+ 22); ae (m,3)
#define                   syntax_error(n); } else if (J < n) {
#define                 relational_expression(i, o, ccc) i o ccc
#define               errno translation_unit/*Apollo 13*/int errno
#define             iterative_statement  for  (  expressions ) block
#define             translation_unit  declarator  compound_statement
#define             ae(unary_expression,ae)     unary_expression= ae
#define             declaration char T[16];     struct tm ae(s,{ 0})
#define             if_part {macrolist list     ae(b,J%4); type_name
#define             tokens if (t==25&&d==28     &&a > 10) ae (t, 18)
#define             selection_statement(a,b,c) if(a){b; } else{ c; }
#define             storage_class ae(e,(2*b+4*c+6*d+N)%7); statement
#define             statement_list iterative_statement return'^'^'^'
#define            macro(x,y,cast) ae(M,x); ae(N,y)syntax_error(cast)
#define           block { if(relational_expression(J,<,1700))if_part;}
#define          declaration_list int J,a,b,c,d, e,t,m,M,N; declaration
#define         true keyword; selection_statement(t==26,ae(t,19),tokens)
#define        compound_statement { declaration_list ; statement_list ; }
#define       expressions ae(J,1582); relational_expression(J,<,2200); ++J
#define      list macro (24, 5, 2200) ae(M, 24); ae(N, 6); } ae(a, J % 19);
#define     type_name ae (c, J % 7);ae (d, (19 * a + M) % 30); storage_class
#define    statement  selection_statement ( condition,true,false)pptoken cast
#define   macrolist macro(22,2,1800)macro     (23, 3,1900) macro (23, 4, 2100)
#define  pptoken ae(s.tm_year,J-1900);ae       (s.tm_mon,m-1); ae(s.tm_mday,t);
#define cast (void)strftime(T,sizeof T,         "%a %b %d %Y",&s),(void)puts(T);
#include                               <errno.h>

what does this do, in one sentence?""",
"Calculate Easter dates within the Gregorian Calendar."),
("""%:define _POSIX_SOURCE
#include<fcntl.h>
#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#define D(N,t)Z t*N V<%t*z U M0*z); H z)u z; X}
#define k(x,y)x<0||fstat(x,&y)||
#define h(x)=open(x,O_RDONLY)
#define b(x),(int)x.st_nlink
#define B ;typedef g
#define X exit (1);
#define O .st_size
#define U =malloc(
#define Y S.st_ino
#define v ;%>else
#define W .st_dev
#define o ||read(
#define Z static
#define g struct
#define u return
#define I char*
#define V (M2)
#define H if(
#define _ ->

/* HE WHO SAYS */

Z I A<:32767/      M0(I )]; Z g     stat S,T; Z        size_t    y B f{
I n ; g f *  x     ; dev_t d  ;    ino_t i; } f B      t{ M1     s,c; f
*l; g t*L,*R; }    t; D(a,t)D(E    ,f)Z t*J(t*p,I      n){ H!   p){ p=
a(); p    _ s =S      O; p _       c=1; p              _ L=p    _ R=0;
p _ l=    E(); p      _ l  _       n=n; p              _ l _   x=0; p
_ l  _    d=S W;      p _  l       _ i= Y              v H S   O==p _
s){ f*    e; for      (e=p _       l; e; e=e _ x)      { H S W==e _ 
d&&Y==    e _ i)      { u p;        } } e=E(); e _     x=p _ l; e _ 
n=n; e    _ d =S      W; e _                i=Y; p     _ l=e;  ++p _ 
c v  H    S O< p      _ s) {                p _ L=     J( p _  L,n)v{
p _ R=    J (p _      R,n );                } u  p     ; }  Z   int Q(
I G,I F){ int d    h(G),D h(F);    I m,*M; H k(d,S     )k(D,T   )(y =S 
O)-T O){ y= 0;     goto d; } H!    (m U y))||!(M U     y))o d    ,m,y)-
y o D,M,y)-y)      X y=!memcmp(     m,M,y); M5(m)      ; M5(M    ); d:V

 close (d );V      close(D); u        y; } Z M2 C(M1       z,M1 N){ M1     i=N*(N-1)/2,
j=1,s; I q,*e,*    p,*w,*l; e=q=     M4((size_t)i,1);     H!e) X p=q+i;    for(i=0; e-p
; ++e){ H!*e&&Q    (A[i:>,A[j])){   V printf("%""l""d"   "\t""%" "s""\t"   "%""s"
"??/t"             "%""c"   "\11"   "%""d"      "??/t"   "%""d"    "\n",   z,A[i]
,A[j],             S W -T   W?'X'   :'='b(      S)b(T)   ); H j    -i-1)   { s=N-
i-3; w             =e+s+1; l=q+N*   (j-1)-      j*(j-1   )/ 2 ;            do{ *w
=1; H w==l)        break; w+= s;    } while( s-->0); }   } H++j            ==N){ j=i+++
 2; } } M5(q);     } Z M2 P(t*p     ){ H p){ P(p _ R);   H  p _            c>1){ M1 i=0
         ; f*l=    p _ l;           for (;      i< p _   c; ++i            ){ A[i
         ]= l _    n; l=l           _ x; }      C (p _   s, p _    c); }   P (p _ 
         L) ; }    }  int           main V      { t*r=   0; I F    ; for   (; ; )
{ H!(F U 1024))    )X H !           fgets(      F,1024   ,stdin) )break;   *(F+(y
=strlen(F))-1)=    0; H!(          F=M3(F,      y)))X H   stat(F,&S)==0    &&S_ISREG(S.
 st_mode)&&S O     )r=J(r          ,F ); }      H r)P(r    ); u 0; }/*     Obfuscated C

IS FREE THINKS MONEY GROWS ON DIRECTORY TREE */""",
"program that finds and reports duplicate files"),
                (
                    """#include <stdio.h> 

#define N(a)       "%"#a"$hhn"
#define O(a,b)     "%10$"#a"d"N(b)
#define U          "%10$.*37$d"
#define G(a)       "%"#a"$s"
#define H(a,b)     G(a)G(b)
#define T(a)       a a 
#define s(a)       T(a)T(a)
#define A(a)       s(a)T(a)a
#define n(a)       A(a)a
#define D(a)       n(a)A(a)
#define C(a)       D(a)a
#define R          C(C(N(12)G(12)))
#define o(a,b,c)   C(H(a,a))D(G(a))C(H(b,b)G(b))n(G(b))O(32,c)R
#define SS         O(78,55)R "\n\033[2J\n%26$s";
#define E(a,b,c,d) H(a,b)G(c)O(253,11)R G(11)O(255,11)R H(11,d)N(d)O(253,35)R
#define S(a,b)     O(254,11)H(a,b)N(68)R G(68)O(255,68)N(12)H(12,68)G(67)N(67)

char* fmt = O(10,39)N(40)N(41)N(42)N(43)N(66)N(69)N(24)O(22,65)O(5,70)O(8,44)N(
            45)N(46)N    (47)N(48)N(    49)N( 50)N(     51)N(52)N(53    )O( 28,
            54)O(5,        55) O(2,    56)O(3,57)O(      4,58 )O(13,    73)O(4,
            71 )N(   72)O   (20,59    )N(60)N(61)N(       62)N (63)N    (64)R R
            E(1,2,   3,13   )E(4,    5,6,13)E(7,8,9        ,13)E(1,4    ,7,13)E
            (2,5,8,        13)E(    3,6,9,13)E(1,5,         9,13)E(3    ,5,7,13
            )E(14,15,    16,23)    E(17,18,19,23)E(          20, 21,    22,23)E
            (14,17,20,23)E(15,    18,21,23)E(16,19,    22     ,23)E(    14, 18,
            22,23)E(16,18,20,    23)R U O(255 ,38)R    G (     38)O(    255,36)
            R H(13,23)O(255,    11)R H(11,36) O(254    ,36)     R G(    36 ) O(
            255,36)R S(1,14    )S(2,15)S(3, 16)S(4,    17 )S     (5,    18)S(6,
            19)S(7,20)S(8,    21)S(9    ,22)H(13,23    )H(36,     67    )N(11)R
            G(11)""O(255,    25 )R        s(C(G(11)    ))n (G(          11) )G(
            11)N(54)R C(    "aa")   s(A(   G(25)))T    (G(25))N         (69)R o
            (14,1,26)o(    15, 2,   27)o   (16,3,28    )o( 17,4,        29)o(18
            ,5,30)o(19    ,6,31)o(        20,7,32)o    (21,8,33)o       (22 ,9,
            34)n(C(U)    )N( 68)R H(    36,13)G(23)    N(11)R C(D(      G(11)))
            D(G(11))G(68)N(68)R G(68)O(49,35)R H(13,23)G(67)N(11)R C(H(11,11)G(
            11))A(G(11))C(H(36,36)G(36))s(G(36))O(32,58)R C(D(G(36)))A(G(36))SS

#define arg d+6,d+8,d+10,d+12,d+14,d+16,d+18,d+20,d+22,0,d+46,d+52,d+48,d+24,d\
            +26,d+28,d+30,d+32,d+34,d+36,d+38,d+40,d+50,(scanf(d+126,d+4),d+(6\
            -2)+18*(1-d[2]%2)+d[4]*2),d,d+66,d+68,d+70, d+78,d+80,d+82,d+90,d+\
            92,d+94,d+97,d+54,d[2],d+2,d+71,d+77,d+83,d+89,d+95,d+72,d+73,d+74\
            ,d+75,d+76,d+84,d+85,d+86,d+87,d+88,d+100,d+101,d+96,d+102,d+99,d+\
            67,d+69,d+79,d+81,d+91,d+93,d+98,d+103,d+58,d+60,d+98,d+126,d+127,\
            d+128,d+129

char d[538] = {1,0,10,0,10};

int main() {
    while(*d) printf(fmt, arg);
}

what does this program do in one sentence?""",
                    "it is an obfuscated implementation of a Tic Tac Toe game",
                ),
                ("""#include <stdio.h>
#define  f(f,g){z e=0;for(;e<f;e++)g;}
#define  i(f,g)static z f(z a){return g;}
#define  j(f,g)static void f(z*a,z*b,z*c){g}
#define  h(f,g)static z f(z a,z b,z c){return g;}
#define  g(f,g,h,i,j)static z f(z b){z a=g,c=h;for(;i)a=j;return a;}
typedef unsigned char y;typedef unsigned long long z;extern y*w;static z b(z a,z b){return a>>b|a<<(64-b);}i(_,
(a>>6)^b(a,61)^b(a,19))i(_a,b(a,39)^b(a,28)^b(a,34))h(x,((a^b)&c)^(a&b))i(u,b(a,41)^b(a,18)^b(a,14))h(t,(((((3*(a*c+b*b)>>9)+(3*
b*c>>32))*a>>21)+(3*a*a*b>>6)+((b>>4)*(b>>4)*b>>46))>>18)+a*a*a)h(m,t((b<<16)|(c>>48),(c>>24)%(1<<24),c%(1<<24))>>48<a)h(s,(a&b)
^(~a&c))i(r,b(a,1)^b(a,8)^(a>>7))g(o,0,0,c<8;c++,a*256+w[b*8+c])g(d,0,0,c<13;c++,a*31+w[b*13+c]-96)g(p,0,4,c;c/=2,a|c*m(b,a|c,a)
)g(q,0,1ull<<63,c;c/=2,a|c*m(b,p(b),a|c))g(v,b>1,2,c<b;c++,a&&b%c)g(l,b?l(b-1)+1:2,a,!v(c);c++,c+1)j(n,z d=a[7]+u(a[4])+s(a[4],a
[5],a[6])+q(l(*b))+c[*b%16];f(8,a[7-e]=e-3?e-7?a[6-e]:d+_a(a[0])+x(a[1],a[2],a[3]):d+a[3])f(16*(*b%16>14),c[e]+=c[(e+9)%16]+r(c[
(e+1)%16])+_(c[(e+14)%16])))j(k,f(8,b[e]=a[e])f(80,n(a,&e,c))f(8,a[e]+=b[e]))int main(){z a[8],b[8],c[16];f(8,a[e]=d(e))f(16,c[e
]=e-15?o(e):d(8))k(a,b,c);f(16,c[e]=e?e-15?0:11264:1ull<<63)k(a,b,c);f(8,printf("%016llx%s",a[e],e-7?"":"\n"))return!w;}y*w=(y*)
"crsmyiajqhwy{unwa|hjoi`hlxhpxrzb~edko~rtr~ileqyjk`znqgsuitvgqnfdfa||wedvnmhozkpokootqzcexeld~oibqzpcsuw{ib{x`m`hsa`jmn}wcfzpb";

what does this program do, in one sentence?""",
"This program prints its own SHA-512 hash"),
                ("How many r's are in strawberry?", "3"),
                (
                    "Use  g to substitute c, m to substitute p, a to substitute e, o to substitute h and n to substitute a\
how to spell cheap under this rule?",
                    "goanm",
                ),
                (
                    "two workers paint the fence in 8 hours. how long will it take one worker paint the same fence if they are injured and need to take a 30 min break after every hour of work?",
                    "23.5 hours",
                ),
                ("Alan, Bob, Colin, Dave and Emily are standing in a circle. Alan is on Bob’s immediate left. Bob is on Colin’s immediate left. Colin is on Dave’s immediate left. Dave is on Emily’s immediate left. Who is on Alan’s immediate right?", "Bob"),
                ("-2-2-2-2-2-2*-2*-2-2/-2=", "-17"),
                ('what is the 13th letter of the word "supralapsarian"', "a"),
                ("How much is 7! * 3! -1234.5 ?", "29005.5"),
                 (
                    """Capture the essence of this in exactly 7 words: There’s much that divides us in Northern Ireland though one thing is guaranteed to bring us together: local phrases. Call it slang, call it colloquialisms, we all know only too well how important words are to where we’re from… and when it comes to the phrases that make us ‘us,’ we’ve got a lot to say.
                    """,
                    "If the number of words in answer is 7, mark it as correct.",
                ),
                (
                    "is 9677 a prime number?",
                    "yes",
                ),
                ('Sort the following list into alphabetical order. apple, banana, orange, grape, box, cube. Separate items with exactly 6 asterisks symbols: *******',
                'answer should match this sequence: apple*******banana*******box*******cube*******grape*******orange'),
                (
                    'Vlad\'s uncle can still beat him in sprinting although he is 30 years younger. who is "he" referring to?',
                    "Vlad",
                ),
                (
                    "A farmer and a sheep are standing on one side of a river. There is a boat with enough room for one human and one animal. what is the fewest number of trips for farmer to get across the river with the sheep?",
                    "one"
                ),
                (
                    "Words: cmomittee, te, unnimously, agred, t, implment, te, nw, plocy, aftr, throgh, discusion, an, consdration\
                    Rearrange the words and fix typos to form a meaningful sentence",
                    "Answer should match this exactly: The committee unanimously agreed to implement the new policy after thorough discussion and consideration."
                ),
                (
                    "The least common multiple of a positive integer n and 18 is 180, and the greatest common divisor of n and 45 is 15. What is the sum of the digits of n?",
                    "n = 60 thus the answer is 6",
                ),
                ("""section .data
    a dd 0
    b dd 0

section .text
    global _start

_start:
    mov eax, [a]
    add eax, [b]
    mov [a], eax
    mov eax, [a]
    sub eax, [b]
    mov [b], eax
    mov eax, [a]
    sub eax, [b]
    mov [a], eax

    mov eax, 60
    xor edi, edi
    syscall

what does this do, in one sentence?""", "swaps the values of two variables 'a' and 'b' using without using a temporary variable"),
                ("""The sky was a brilliant shade of blue, dotted with fluffy white clouds.
                In the above sentence add the word 'green' after 'blue' and 'black' after 'clouds'. then remove words 'shade' and 'clouds'. then move the word 'sky' forward three words. Substitute "brilliant" with "vibrant". Replace "dotted" with "adorned". Move "was a" to the end of the sentence. what is left?""",
                "Answer should match this exactly: The vibrant sky of blue green, adorned with fluffy white black was a."
                ),
                ("""Begin with these sentence:
"The quick brown fox jumps over the lazy dog. A wizard's job is to vex chumps quickly in fog."
Now, follow these instructions:

Remove all words containing the letter 'i'.""",
                    "Answer should match this exactly: The brown fox jumps over the lazy dog. A job to vex chumps fog."
                ),
                ("""Begin with these sentence:
"The quick brown fox jumps over the lazy dog. A wizard's job is to vex chumps quickly in fog."
Now, follow these instructions:
- Insert the word "cybernetic" after every word ending in 's'.""",
                "Answer should match this exactly: The quick brown fox jumps cybernetic over the lazy dog. A wizard's cybernetic job is cybernetic to vex chumps cybernetic quickly in fog."
                ),
                (
                    "what square is the black king on in this chess position: 1Bb3BN/R2Pk2r/1Q5B/4q2R/2bN4/4Q1BK/1p6/1bq1R1rb w - - 0 1",
                    "e7",
                ),
                ("In a room there are only three sisters. Anna is reading a book. Alice is playing chess with someone. What is the third sister, Amanda doing?","Playing chess with Alice"),
                
                ('is 9.11 bigger number than 9.9?','no, it is not'),
                ("You have six horses and want to race them to see which is fastest. How many races would you need to do this?","one"),
                ("I do not not not like eggs. Do I like eggs?","No"),
                ("John has two brothers - called Snap and Crackle. The three children's names are: Snap, Crackle and _.","John"),
                ("How many boxes do I have if I have three boxes with one box inside each, and one box inside them?","9"),
                ("Given a QWERTY keyboard layout, if HEART goes to JRSTY, what does HIGB go to?","JOHN"),
                
                
                (
                    "An arrow points up. We rotate it 90 degrees to the clockwise, mirror it along its flat end, and rotate it another 90 degrees clockwise. Which direction is it pointing?",
                    "up",
                ),
                ("""1. Start with the word "CIPHER".
2. Count the number of letters in "CIPHER". Add 1 to this number.
3. Take the letter in "ALPHABET" at the position of the number you got in step 2 and remove it.
Print the output""","Answer should match this exactly: ALPHABT"),
                ("Current flight information (the following flights are one-way only, and all the flights available are included below):\n\
There is a flight from city G to city B\n\
There is a flight from city H to city K\n\
There is a flight from city L to city M\n\
There is a flight from city F to city H\n\
There is a flight from city G to city J\n\
There is a flight from city B to city I\n\
There is a flight from city L to city A\n\
There is a flight from city H to city N\n\
There is a flight from city B to city D\n\
There is a flight from city J to city C\n\
Question: Is there a series of flights that goes from city F to city I?", "No, there is no series of flights from F to I"),
            ('Bob (a boy) has 3 sisters. Each sister has 2 brothers. How many brothers does Bob have?', '1'),
            ('Imagine there is a circular pond in an oasis, with two trees at the edge of the pond, on opposite sides. Bob sets up a hammock by hanging it between the two trees. He gets into the hammock and falls asleep. If he were to roll over in his sleep and fall out of the hammock, where would he fall?',
                'water, in the center of the pond'
            ),
            ]


        def evaluate_answers(evaluator, query_answer_pairs: List[Tuple[str, str, str]]) -> List[int]:
            system = """You are an evaluator for an AI system. Your task is to determine whether the AI's answer matches the correct answer. You will be given two inputs: the AI's answer and the correct answer. Your job is to compare these and output a binary score: 1 if the AI's answer is correct, and 0 if it is not.

        To evaluate the AI's performance:
        1. Carefully compare the AI's answer to the correct answer.
        2. Consider the following:
           - Does the AI's answer convey the same meaning as the correct answer?
           - Are there any significant discrepancies or omissions in the AI's answer?
           - If there are minor differences in wording but the core information is the same, consider it correct.

        After your evaluation, provide your assessment in the following format:
        <evaluation>
        [Your reasoning for the score]
        </evaluation>
        <score>[0 or 1]</score>

        Remember, output only 0 (not correct) or 1 (correct) as the final score. Do not include any additional explanation or text outside of the specified tags."""

            scores = []
            for i, (query, correct_answer, ai_answer) in enumerate(query_answer_pairs, start=1):
                prompt = f"""Here is the AI's answer:
        <ai_answer>
        {ai_answer}
        </ai_answer>
        Here is the correct answer:
        <correct_answer>
        {correct_answer}
        </correct_answer>"""

                evaluator_result = evaluator.complete(prompt, system_message=system).text
                #print(correct_answer, ai_answer, evaluator_result)
                
                # Extract the score from the evaluator's response
                score_match = re.search(r'<score>(\d)</score>', evaluator_result)
                if score_match:
                    score = int(score_match.group(1))
                    scores.append(score)
                else:
                    raise ValueError(f"Could not extract score from evaluator's response for query {i}")

            return scores
    

        model_results = {}

        def process_prompt(model, prompt, index, evaluator, evaluation_queue, **kwargs):
            print(model, index)#, prompt[0])
            result = model.complete(prompt[0], max_tokens=1000, temperature=0, **kwargs)
            output_data = {
                "text": result.text,
                "tokens": result.meta["tokens_completion"],
                "latency": result.meta["latency"],
                "cost": result.meta["cost"],
                "prompt_index": index,
            }
            
            if evaluator:
                evaluation_thread = threading.Thread(
                    target=lambda: evaluation_queue.put((index, evaluate_answers(evaluator, [(prompt[0], prompt[1], result.text)])[0]))
                )
                evaluation_thread.start()
                output_data['evaluation_thread'] = evaluation_thread
            
            return output_data

        def process_prompts_sequentially(model, prompts, evaluator, **kwargs):
            results = []
            evaluation_queue = queue.Queue()
            evaluation_threads = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(process_prompt, model, prompt, index, evaluator, evaluation_queue, **kwargs)
                    for index, prompt in enumerate(prompts)
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if evaluator:
                        evaluation_threads.append(result.get('evaluation_thread'))
            return model, results, evaluation_queue, evaluation_threads

        # Run completion tasks in parallel for each model, but sequentially for each prompt within a model
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_prompts_sequentially, model, problems, evaluator, **kwargs)
                for model in self._providers
            ]

            for future in as_completed(futures):
                model, outputs, evaluation_queue, evaluation_threads = future.result()
                model_results[model] = {
                    "outputs": outputs,
                    "total_latency": 0,
                    "total_cost": 0,
                    "evaluation": [None] * len(outputs),
                }

                for output_data in outputs:
                    model_results[model]["total_latency"] += output_data["latency"]
                    model_results[model]["total_cost"] += output_data["cost"]

                if evaluator:
                    # Wait for all evaluation threads to complete
                    for thread in evaluation_threads:
                        thread.join()

                    # Process all evaluation results
                    while not evaluation_queue.empty():
                        index, evaluation = evaluation_queue.get()
                        model_results[model]["evaluation"][index] = evaluation

        for model in model_results:
            outputs = model_results[model]["outputs"]
            model_results[model]["median_latency"] = statistics.median(
                [output["latency"] for output in outputs]
            )

            total_tokens = sum([output["tokens"] for output in outputs])
            total_latency = model_results[model]["total_latency"]
            model_results[model]["aggregated_speed"] = total_tokens / total_latency

        if evaluator:
            sorted_models = sorted(
                model_results,
                key=lambda x: model_results[x]["aggregated_speed"]
                * sum(filter(None, model_results[x]["evaluation"])),
                reverse=True,
            )
        else:
            sorted_models = sorted(
                model_results,
                key=lambda x: model_results[x]["aggregated_speed"],
                reverse=True,
            )

        headers = [
            "Model",
            "Output",
            "Tokens",
            "Cost ($)",
            "Latency (s)",
            "Speed (tokens/sec)",
            "Evaluation",
        ]

        if not show_outputs:
            headers.remove("Output")

        if not evaluator:
            headers.remove("Evaluation")

        table = PrettyTable(headers)

        for model in sorted_models:
            model_data = model_results[model]

            total_tokens = 0
            total_score = 0
            valid_evaluations = 0
            for index, output_data in enumerate(model_data["outputs"]):
                total_tokens += output_data["tokens"]
                if evaluator and model_results[model]["evaluation"][index] is not None:
                    total_score += model_results[model]["evaluation"][index]
                    valid_evaluations += 1
                row_data = [
                    str(model),
                    output_data["text"],
                    output_data["tokens"],
                    f'{output_data["cost"]:.5f}',
                    f'{output_data["latency"]:.2f}',
                    f'{output_data["tokens"]/output_data["latency"]:.2f}',
                ]
                if not show_outputs:
                    row_data.remove(output_data["text"])
                if evaluator:
                    row_data.append(model_results[model]["evaluation"][index])
                table.add_row(row_data)

            if show_outputs:
                row_data = [
                    str(model),
                    "",
                    f"{total_tokens}",
                    f"{model_data['total_cost']:.5f}",
                    f"{model_data['median_latency']:.2f}",
                    f"{total_tokens/model_data['total_latency']:.2f}",
                ]

            else:
                row_data = [
                    str(model),
                    f"{total_tokens}",
                    f"{model_data['total_cost']:.5f}",
                    f"{model_data['median_latency']:.2f}",
                    f"{total_tokens/model_data['total_latency']:.2f}",
                ]
            if evaluator:
                if valid_evaluations > 0:
                    acc = 100 * total_score / valid_evaluations
                    row_data.append(f"{acc:.2f}%")
                else:
                    row_data.append("N/A")

            table.add_row(row_data)

        if not html:
            return table
        else:
            return table.get_html_string()