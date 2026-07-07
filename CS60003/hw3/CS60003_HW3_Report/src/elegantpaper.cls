% Author: Dongsheng Deng
% Homepage: https://ddswhu.me/
% Email: elegantlatex2e@gmail.com
% Lastest Version: https://github.com/ElegantLaTeX/ElegantPaper

% --- Class structure: identification part
\ProvidesClass{elegantpaper}[2022/12/31 v0.11 ElegantLaTeX Paper class]
\NeedsTeXFormat{LaTeX2e} 

%%%
\RequirePackage{kvoptions}
\RequirePackage{etoolbox}
\RequirePackage{calc}

\SetupKeyvalOptions{family=ELEGANT, prefix=ELEGANT@, setkeys=\kvsetkeys}
\newcommand{\ekv}[1]{\kvsetkeys{ELEGANT}{#1}}

\DeclareStringOption[en]{lang}
\DeclareVoidOption{cn}{\ekv{lang=cn}}
\DeclareVoidOption{en}{\ekv{lang=en}}

\DeclareStringOption[cm]{math}

\DeclareStringOption[numeric-comp]{citestyle}
\DeclareStringOption[numeric]{bibstyle}

\DeclareStringOption[biber]{bibend}
\DeclareVoidOption{biber}{\ekv{bibend=biber}}
\DeclareVoidOption{bibtex}{\ekv{bibend=bibtex}}


\DeclareStringOption[ctexfont]{chinesefont}
\DeclareVoidOption{ctexfont}{\ekv{chinesefont=ctexfont}}
\DeclareVoidOption{founder}{\ekv{chinesefont=founder}}
\DeclareVoidOption{nofont}{\ekv{chinesefont=nofont}}

\DeclareStringOption[listings]{code}
\DeclareStringOption[light]{codebg}

\DeclareVoidOption{newtx}{\ekv{math=newtx}}
\DeclareVoidOption{mtpro2}{\ekv{math=mtpro2}}
\DeclareVoidOption{cm}{\ekv{math=cm}}

\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessKeyvalOptions*\relax
\LoadClass{article}

\RequirePackage{hyperref}
\hypersetup{
  pdfborder={0 0 0},
  colorlinks=true,
  linkcolor={winered},
  urlcolor={winered},
  filecolor={winered},
  citecolor={winered},
  linktoc=all,
}
% settings for the hyperref and geometry
\RequirePackage[
  left=1in,
  right=1in,
  top=0.31in,
  bottom=1in,
  headheight=0pt,
  headsep=0pt]{geometry}
\setlength{\headsep}{5pt}

\RequirePackage{amsthm}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{indentfirst}
\RequirePackage{booktabs}
\RequirePackage{multicol}
\RequirePackage{multirow}
% \RequirePackage{linegoal}
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{fancyvrb}
\RequirePackage{abstract}
\RequirePackage{hologo}

\linespread{1.35}
\graphicspath{{../pic/}{pic/}{image/}{figure/}{fig/}{img/}}

% caption settings 
\RequirePackage[labelfont={bf}]{caption}
\captionsetup[table]{skip=3pt}
\captionsetup[figure]{skip=3pt}

% list/itemize/enumerate setting
\RequirePackage[shortlabels,inline]{enumitem}
\setlist{nolistsep}


% define the hyperref color 
\RequirePackage{xcolor}
\definecolor{winered}{rgb}{0.184,0.573,0.894}

\definecolor{lightgrey}{rgb}{0.902,0.902,0.902}
\definecolor{commentcolor}{RGB}{0,146,127}
\definecolor{frenchplum}{RGB}{93,31,185}
\definecolor{topaztitle}{RGB}{8,55,125}
\definecolor{topazsection}{RGB}{0,79,168}
\definecolor{topazsubsection}{RGB}{14,99,185}
\definecolor{topazsubsubsection}{RGB}{52,130,197}
\ifdefstring{\ELEGANT@codebg}{dark}{
  \definecolor{codebg}{RGB}{41,45,62}
  \definecolor{codetext}{RGB}{212,212,212}
  \definecolor{codekeyword}{HTML}{C792EA}
  \definecolor{codecommand}{HTML}{82AAFF}
  \definecolor{codestring}{HTML}{C3E88D}
  \definecolor{codenumber}{HTML}{FF5370}
  \definecolor{codecomment}{HTML}{676E95}
  \definecolor{codeemph}{HTML}{89DDFF}
  \definecolor{codefunction}{HTML}{82AAFF}
  \definecolor{codebuiltin}{HTML}{FFCB6B}
  \definecolor{codetype}{HTML}{DECB6B}
}{%
  % light background, higher contrast for print/PDF
  \definecolor{codebg}{HTML}{F1F1F1}
  \definecolor{codetext}{HTML}{0E0E0E}
  \definecolor{codekeyword}{HTML}{5D1FB9}
  \definecolor{codecommand}{HTML}{3B84E4}
  \definecolor{codestring}{HTML}{068A5E}
  \definecolor{codenumber}{HTML}{E95D00}
  \definecolor{codecomment}{HTML}{6A737D}
  \definecolor{codeemph}{HTML}{0083BB}
  \definecolor{codefunction}{HTML}{0083BB}
  \definecolor{codebuiltin}{HTML}{3B84E4}
  \definecolor{codetype}{HTML}{5D1FB9}
}
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\color{topazsection}\normalfont\Large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\color{topazsubsection}\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\color{topazsubsubsection}\normalfont\normalsize\bfseries}}


% add the email cmd 
\newcommand\email[1]{\href{mailto:#1}{\nolinkurl{#1}}}

% font settings


\PassOptionsToPackage{no-math}{fontspec}
\PassOptionsToPackage{quiet}{fontspec}
\RequirePackage{iftex}

\ifXeTeX
  \RequirePackage[no-math]{fontspec}
  \setmainfont{texgyretermes}[
    UprightFont = *-regular ,
    BoldFont = *-bold ,
    ItalicFont = *-italic ,
    BoldItalicFont = *-bolditalic ,
    Extension = .otf ,
    Scale = 1.0]
    
  \setsansfont{texgyreheros}[
    UprightFont = *-regular ,
    BoldFont = *-bold ,
    ItalicFont = *-italic ,
    BoldItalicFont = *-bolditalic ,
    Extension = .otf ,
    Scale = 0.9]
\else
  \RequirePackage{newtxtext}
  \RequirePackage[scaled=.90]{helvet}
\fi

\ifdefstring{\ELEGANT@lang}{cn}{
  \newcommand*\ELEGANT@ctexfontset{auto}
  \ifXeTeX
    \IfFontExistsTF{PingFang SC}{%
      \renewcommand*\ELEGANT@ctexfontset{macnew}%
    }{%
      \IfFontExistsTF{STHeiti}{%
        \renewcommand*\ELEGANT@ctexfontset{macold}%
      }{%
        \renewcommand*\ELEGANT@ctexfontset{fandol}%
      }%
    }%
  \fi

  \ifdefstring{\ELEGANT@chinesefont}{founder}{
    \RequirePackage[UTF8,scheme=plain,fontset=none]{ctex}
    \setCJKmainfont[BoldFont={FZHei-B01},ItalicFont={FZKai-Z03}]{FZShuSong-Z01}
    \setCJKsansfont[BoldFont={FZHei-B01}]{FZKai-Z03}
    \setCJKmonofont[BoldFont={FZHei-B01}]{FZFangSong-Z02}
    \setCJKfamilyfont{zhsong}{FZShuSong-Z01}
    \setCJKfamilyfont{zhhei}{FZHei-B01}
    \setCJKfamilyfont{zhkai}[BoldFont={FZHei-B01}]{FZKai-Z03}
    \setCJKfamilyfont{zhfs}[BoldFont={FZHei-B01}]{FZFangSong-Z02}
    \newcommand*{\songti}{\CJKfamily{zhsong}}
    \newcommand*{\heiti}{\CJKfamily{zhhei}}
    \newcommand*{\kaishu}{\CJKfamily{zhkai}}
    \newcommand*{\fangsong}{\CJKfamily{zhfs}}}{\relax}
  
  \ifdefstring{\ELEGANT@chinesefont}{nofont}{
    \RequirePackage[UTF8,scheme=plain,fontset=none]{ctex}}{\relax}

  \ifdefstring{\ELEGANT@chinesefont}{ctexfont}{
    \RequirePackage[UTF8,scheme=plain,fontset=\ELEGANT@ctexfontset]{ctex}}{\relax}
  
  \AfterEndPreamble{
    \setlength\parindent{2\ccwd}}
}{\relax}


\ifcsname kaishu\endcsname
  \newcommand{\citshape}{\kaishu}
\else
  \newcommand{\citshape}{\itshape}
\fi
\ifcsname kaishu\endcsname
  \newcommand{\cnormal}{\kaishu}
\else
  \newcommand{\cnormal}{\normalfont}
\fi

\ifcsname fangsong\endcsname
  \newcommand{\cfs}{\fangsong}
\else
  \newcommand{\cfs}{\normalfont}
\fi

\ifdefstring{\ELEGANT@math}{newtx}{
  \RequirePackage[noOT1]{newtxmath}
  \let\Bbbk\relax
  \RequirePackage{esint}
  %%% use yhmath pkg, uncomment following code
  % \let\oldwidering\widering
  % \let\widering\undefined
  % \RequirePackage{yhmath}
  % \let\widering\oldwidering

  %%% use esvect pkg, uncomment following code
  % \RequirePackage{esvect}

  \DeclareSymbolFont{CMlargesymbols}{OMX}{cmex}{m}{n}
  \let\sumop\relax\let\prodop\relax
  \DeclareMathSymbol{\sumop}{\mathop}{CMlargesymbols}{"50}
  \DeclareMathSymbol{\prodop}{\mathop}{CMlargesymbols}{"51}
}{\relax}


\ifdefstring{\ELEGANT@bibend}{bibtex}{%
  % biblatex emits a fixed fallback warning when using the bibtex backend.
  % Keep dual-backend support, but silence this one expected warning only.
  \RequirePackage{silence}
  \WarningFilter{biblatex}{Using fall-back bibtex backend}%
}{}

\RequirePackage[
  backend=\ELEGANT@bibend,
  citestyle=\ELEGANT@citestyle,
  bibstyle=\ELEGANT@bibstyle]{biblatex}

\RequirePackage{appendix}

\ifdefstring{\ELEGANT@lang}{cn}{
  \theoremstyle{plain}% default
  \newtheorem{theorem}{定理}[section] %
  \newtheorem{lemma}[theorem]{引理} %
  \newtheorem{proposition}[theorem]{命题} %
  \newtheorem*{corollary}{推论} %
  \theoremstyle{definition} %
  \newtheorem{definition}{定义}[section] %
  \newtheorem{conjecture}{猜想}[section] %
  \newtheorem{example}{例}[section] %
  \theoremstyle{remark} %
  \newtheorem*{remark}{\normalfont\bfseries 评论} %
  \newtheorem*{note}{\normalfont\bfseries 注} %
  \newtheorem{case}{\normalfont\bfseries 案例} %
  \renewcommand*{\proofname}{\normalfont\bfseries 证明} %
  \renewcommand\contentsname{目录}
  \renewcommand\refname{参考文献} %
  \renewcommand\figurename{图} %
  \renewcommand\tablename{表}%
  \newcommand\versiontext{版本：}%
  \newcommand\updatetext{日期：}%
  \newcommand{\ebibname}{参考文献}
  \renewcommand\abstractname{摘\hspace{2em}要}
  \setlength\parindent{2\ccwd}
  \renewcommand{\abstracttextfont}{\small\citshape\noindent\ignorespaces}
  % 新定义命令
  \newcommand{\keywords}[1]{\vskip2ex\par\noindent\normalfont{\bfseries 关键词: } #1}
  \newcommand\figref[1]{{\bfseries 图~\ref{#1}}}
  \newcommand\tabref[1]{{\bfseries 表~\ref{#1}}}
  \renewcommand{\appendixtocname}{附录}
  \renewcommand{\appendixpagename}{附录}}{\relax}

\ifdefstring{\ELEGANT@lang}{en}{
  \theoremstyle{plain}% default
  \newtheorem{theorem}{Theorem}[section] %
  \newtheorem{lemma}[theorem]{Lemma} %
  \newtheorem{proposition}[theorem]{Proposition} %
  \newtheorem*{corollary}{Corollary} %
  \theoremstyle{definition} %
  \newtheorem{definition}{Definition}[section] %
  \newtheorem{conjecture}{Conjecture}[section] %
  \newtheorem{example}{Example}[section] %
  \theoremstyle{remark} %
  \newtheorem*{remark}{Remark} %
  \newtheorem*{note}{Note} %
  \newtheorem{case}{Case} %
  \renewcommand*{\proofname}{\normalfont\bfseries Proof}%
  \newcommand\versiontext{\itshape Version: }%
  \newcommand\updatetext{\itshape Date: }%
  \setlength\parindent{2em}
  \renewcommand{\abstracttextfont}{\sffamily\small\noindent\ignorespaces}
  % newcommands defined in this template.
  \newcommand{\keywords}[1]{\vskip 2ex\par\noindent\normalfont{\bfseries Keywords: } #1}
  \newcommand\figref[1]{{\bfseries Figure~\ref{#1}}}
  \newcommand\tabref[1]{{\bfseries Table~\ref{#1}}}
  \newcommand{\ebibname}{Bibliography}}{\relax}


\def\bibfont{\footnotesize}

\AtBeginEnvironment{verbatim}{\microtypesetup{activate=false}}
\BeforeBeginEnvironment{tabular}{\small}
\AfterEndEnvironment{tabular}{}

  \AtBeginDocument{
  \setlength{\abovedisplayskip}{3pt}
  \setlength{\belowdisplayskip}{3pt}
  \RequirePackage[flushmargin]{footmisc}
  \setlength{\footnotesep}{12pt}
}

\def\IfEmpty#1{%
  \edef\1{\the#1}
  \ifx\1\empty
}
\newtoks\version
\newtoks\institute

\renewcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@0.1em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@ELEGANT@version\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  \global\let\ELEGANT@version\relax
}
\def\@maketitle{%
  \newpage
  \null
  \vskip 0em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE\bfseries \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author\\[1ex]
      \end{tabular}\par}
      \the\institute%
\vskip 0.5ex%
\IfEmpty\version
\else 
{\small\normalfont\citshape\versiontext\the\version}
\fi
\ifx\@date\empty
\else
\vskip 0.1em%
 {\small\normalfont\citshape\updatetext\@date}%
\fi
  \end{center}%
  \par
}
  

\RequirePackage{listings}
\renewcommand{\ttdefault}{cmtt}
\newcommand{\elegantcodefont}{\ttfamily}

\ifdefstring{\ELEGANT@lang}{cn}{%
  \ifdefstring{\ELEGANT@chinesefont}{ctexfont}{%
    \ifcsname newCJKfontfamily\endcsname
      \IfFontExistsTF{Sarasa Mono SC}{%
        \newCJKfontfamily\elegantcjkcodefont[AutoFakeSlant=0.2]{Sarasa Mono SC}%
        \renewcommand{\elegantcodefont}{\ttfamily\elegantcjkcodefont}%
      }{%
        \IfFontExistsTF{Hiragino Sans GB}{%
          \newCJKfontfamily\elegantcjkcodefont[AutoFakeSlant=0.2]{Hiragino Sans GB}%
          \renewcommand{\elegantcodefont}{\ttfamily\elegantcjkcodefont}%
        }{%
          \IfFontExistsTF{PingFang SC}{%
            \newCJKfontfamily\elegantcjkcodefont[AutoFakeSlant=0.2]{PingFang SC}%
            \renewcommand{\elegantcodefont}{\ttfamily\elegantcjkcodefont}%
          }{}%
        }%
      }%
    \fi
  }{\relax}%
}{\relax}

\lstdefinestyle{estyle}{
  basicstyle=%
    \elegantcodefont
    \lst@ifdisplaystyle\footnotesize\fi
}
\lstset{basicstyle=\scriptsize\elegantcodefont\color{codetext},style=estyle}

\lstset{language=[LaTeX]TeX,
		texcsstyle=*\color{codecommand},
		numbers=none,
		breaklines=true,
		keywordstyle=\color{codekeyword},
	frame=tlbr,framesep=4pt,framerule=0pt,
	commentstyle=\color{codecomment},
	stringstyle=\color{codestring},
	emph={elegantpaper,fontenc,fontspec,xeCJK,FiraMono,xunicode,newtxmath,figure,fig,image,img,table,itemize,enumerate,newtxtext,newtxtt,ctex,microtype,description,times,newtx,booktabs,tabular,PDFLaTeX,XeLaTeX,type1cm,BibTeX,cite,gbt7714,lang,hyperref,geometry,amsmath,amssymb,amsthm,mathtools,siunitx,tikz,pgfplots,cleveref,caption,subcaption,graphicx,xcolor,listings,biblatex,csquotes},
	emphstyle={\color{codeemph}},
	morekeywords={DeclareSymbolFont,SetSymbolFont,toprule,midrule,bottomrule,institute,version,includegraphics,setmainfont,setsansfont,setmonofont,setCJKmainfont,setCJKsansfont,setCJKmonofont,setCJKfamilyfont,setCJKfallbackfamilyfont,addCJKfontfeatures,CJKfamily,setCJKfamily,RequirePackage,figref,tabref,email,maketitle,keywords,zhdate,zhtoday,newcommand,renewcommand,providecommand,DeclareRobustCommand,NewDocumentCommand,RenewDocumentCommand,ProvideDocumentCommand,DeclareMathOperator,DeclareMathOperator*,newenvironment,renewenvironment,DeclarePairedDelimiter,DeclarePairedDelimiterX},
	tabsize=2,
	backgroundcolor=\color{codebg}
}

\lstdefinelanguage{python}[]{Python}{
  morekeywords=[2]{self,True,False,None},
  keywordstyle=[2]\color{codeemph},
  moredelim=[is][\color{codefunction}]{def }{(},
  moredelim=[is][\color{codefunction}]{class }{(},
  moredelim=[is][\color{codefunction}]{class }{:},
}

\ifdefstring{\ELEGANT@code}{minted}{
  \RequirePackage{shellesc}
  \ifnum\ShellEscapeStatus>0\relax
		    \RequirePackage{minted}
		    \setminted{
		      breaklines,
		      fontsize=\normalsize,
		      formatcom=\elegantcodefont,
		      bgcolor=codebg,
		      style=default
			    }
		    \setmintedinline{
		      fontsize=\normalsize,
		      formatcom=\elegantcodefont
		    }
		    \makeatletter
		    \apptocmd{\minted@patch@PygmentsStyledef}{%
		      \expandafter\def\csname PYG@tok@c\endcsname{\let\PYG@it=\textit\def\PYG@tc##1{\textcolor{codecomment}{##1}}}%
		      \expandafter\def\csname PYG@tok@c1\endcsname{\let\PYG@it=\textit\def\PYG@tc##1{\textcolor{codecomment}{##1}}}%
	      \expandafter\def\csname PYG@tok@cm\endcsname{\let\PYG@it=\textit\def\PYG@tc##1{\textcolor{codecomment}{##1}}}%
	      \expandafter\def\csname PYG@tok@cp\endcsname{\let\PYG@it=\textit\def\PYG@tc##1{\textcolor{codecomment}{##1}}}%
	      \expandafter\def\csname PYG@tok@k\endcsname{\def\PYG@tc##1{\textcolor{codekeyword}{##1}}}%
	      \expandafter\def\csname PYG@tok@kc\endcsname{\def\PYG@tc##1{\textcolor{codekeyword}{##1}}}%
	      \expandafter\def\csname PYG@tok@kd\endcsname{\def\PYG@tc##1{\textcolor{codekeyword}{##1}}}%
	      \expandafter\def\csname PYG@tok@kn\endcsname{\def\PYG@tc##1{\textcolor{codekeyword}{##1}}}%
	      \expandafter\def\csname PYG@tok@kp\endcsname{\def\PYG@tc##1{\textcolor{codekeyword}{##1}}}%
	      \expandafter\def\csname PYG@tok@kr\endcsname{\def\PYG@tc##1{\textcolor{codekeyword}{##1}}}%
	      \expandafter\def\csname PYG@tok@kt\endcsname{\def\PYG@tc##1{\textcolor{codetype}{##1}}}%
	      \expandafter\def\csname PYG@tok@na\endcsname{\def\PYG@tc##1{\textcolor{codeemph}{##1}}}%
	      \expandafter\def\csname PYG@tok@nb\endcsname{\def\PYG@tc##1{\textcolor{codebuiltin}{##1}}}%
	      \expandafter\def\csname PYG@tok@bp\endcsname{\def\PYG@tc##1{\textcolor{codebuiltin}{##1}}}%
	      \expandafter\def\csname PYG@tok@nc\endcsname{\def\PYG@tc##1{\textcolor{codetype}{##1}}}%
	      \expandafter\def\csname PYG@tok@nf\endcsname{\def\PYG@tc##1{\textcolor{codefunction}{##1}}}%
	      \expandafter\def\csname PYG@tok@nn\endcsname{\def\PYG@tc##1{\textcolor{codetype}{##1}}}%
	      \expandafter\def\csname PYG@tok@no\endcsname{\def\PYG@tc##1{\textcolor{codenumber}{##1}}}%
	      \expandafter\def\csname PYG@tok@nd\endcsname{\def\PYG@tc##1{\textcolor{codeemph}{##1}}}%
	      \expandafter\def\csname PYG@tok@ni\endcsname{\def\PYG@tc##1{\textcolor{codeemph}{##1}}}%
	      \expandafter\def\csname PYG@tok@ne\endcsname{\def\PYG@tc##1{\textcolor{codenumber}{##1}}}%
	      \expandafter\def\csname PYG@tok@nl\endcsname{\def\PYG@tc##1{\textcolor{codeemph}{##1}}}%
	      \expandafter\def\csname PYG@tok@nt\endcsname{\def\PYG@tc##1{\textcolor{codecommand}{##1}}}%
	      \expandafter\def\csname PYG@tok@nv\endcsname{\def\PYG@tc##1{\textcolor{codetext}{##1}}}%
	      \expandafter\def\csname PYG@tok@o\endcsname{\def\PYG@tc##1{\textcolor{codeemph}{##1}}}%
	      \expandafter\def\csname PYG@tok@ow\endcsname{\def\PYG@tc##1{\textcolor{codeemph}{##1}}}%
	      \expandafter\def\csname PYG@tok@p\endcsname{\def\PYG@tc##1{\textcolor{codetext}{##1}}}%
	      \expandafter\def\csname PYG@tok@s\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@s1\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@s2\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@sa\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@sb\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@sc\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@sd\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@se\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@sh\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@si\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@sr\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@ss\endcsname{\def\PYG@tc##1{\textcolor{codestring}{##1}}}%
	      \expandafter\def\csname PYG@tok@mi\endcsname{\def\PYG@tc##1{\textcolor{codenumber}{##1}}}%
	      \expandafter\def\csname PYG@tok@mf\endcsname{\def\PYG@tc##1{\textcolor{codenumber}{##1}}}%
	      \expandafter\def\csname PYG@tok@mh\endcsname{\def\PYG@tc##1{\textcolor{codenumber}{##1}}}%
	      \expandafter\def\csname PYG@tok@il\endcsname{\def\PYG@tc##1{\textcolor{codenumber}{##1}}}%
	      \expandafter\def\csname PYG@tok@m\endcsname{\def\PYG@tc##1{\textcolor{codenumber}{##1}}}%
	      \expandafter\def\csname PYG@tok@err\endcsname{\def\PYG@tc##1{\textcolor{codenumber}{##1}}}%
	    }{}{}
	    \makeatother
		    \newminted{python}{breaklines,fontsize=\normalsize,bgcolor=codebg}
		    \newminted{tex}{breaklines,fontsize=\normalsize,bgcolor=codebg}
	    \newminted{text}{breaklines,fontsize=\normalsize,bgcolor=codebg}
	  \else
	    \PackageWarning{elegantpaper}{Option `code=minted' requires -shell-escape; falling back to listings}
	  \fi
}{\relax}



% restore the tt default family to lmodern tt family
\renewcommand\ttdefault{lmtt}
