# 🔍 What is Git?
**git**은 **VCS(Version Control System,분산 버전 관리 시스템)중** 하나로, 코드 뿐만 아니라 텍스트 디자인 레이아웃 파일 모두 버전관리가 가능하다.  
이를 통해 개발자들은 프로젝트의 변경 사항을 기록하고, 특정 시점의 버전으로 언제든 돌아갈 수 있다.   
이런 버전 관리 시스템은 많은 사람들이 효율적으로 함께 작업하고, 프로젝트를 중심으로 협업할 때 사용할 수 있다.   
각 개발자가 자신만의 프로젝트 버전을 본인 컴퓨터에 갖게된다. 나중에 이러한 개별 버전의 프로젝트를 병합하여 기준이 되는 버전의 프로젝트에 적용 할 수 있게 된다.  
VCS에는 두 종류가 있다.  
## 1. Centralized Version Control
![](https://velog.velcdn.com/images/tiger/post/e237ad92-8187-4d7b-8cac-cae4e2b4d901/image.png)

**각 개발자가 히스토리를 서버에 업데이트하여 즉각적인 동기화가 이루어지는 구조**이다.  
**문제점**은 서버에 문제가 생기면 일을 할 수 없고, 오프라인에서도 또한 일을 못한다는 것이다.  

## 2. Distributed Version Control
![](https://velog.velcdn.com/images/tiger/post/9d02819b-8a7c-431b-91f2-706ed1b0058b/image.png)
Centralized Version Control의 단점을 개선하기 위한 구조가 바로 Distributed Version Control이다.  
CVC처럼 Server에만 히스토리 정보가 있는 것이 아니라, **모든 개발자들이 동일한 히스토리 정보를 가지고 있는 구조**이다.  
![](https://velog.velcdn.com/images/tiger/post/0ce9d11f-effe-4ca7-8b63-5948fd88d06e/image.png)
  
고로, 서버에 문제가 생기거나 다운이 되어도 각 개발자가 동일한 히스토리를 가지고 있기에 서로 정보를 이용해 서버를 복원하고 일을 이어나갈 수 있다. 또한 오프라인에서도 일을 할 수 있다.  
  
여기서 server는 회사내부에서만 사용하는 private server를 사용하는 경우도 있고 Github, Bitbucket같은 클라우드를 이용하는 경우도 있다.  
# 🔍 Git Start
## vscode 연결방법
터미널에서 `code . `하면 열린다.  

## git 명령의 구조  
``` 
git 명령어 -option
git commit -m

git 명령어 --속성
git config --global
```
`git 명령어 --h`를 터미널에서 실행하면 해당 명령어에 대한 전체 옵션과 속성에 대한 간단한 정보를 확인할 수 있다.  
더 자세한 설명은 이 [Git 링크](https://git-scm.com/docs)에서 확인이 가능한다.  

# 🔍 workflow와 함께하는 Git 명령어
### git init 
![](https://velog.velcdn.com/images/tiger/post/23ff2f66-78fe-440b-854c-8f47d57a0ebc/image.png)
  
`git init`명령어는 나의 local 프로젝트 폴더 내에 숨겨진 .git 디렉토리를 생성한다. 이제 Git은 현재 저장소에 대한 모든 변경사항을 추적/관리하게 된다.  
(파일이나 폴더 앞에 .이 붙으면 그 파일/폴더는 숨겨진 파일/폴더이다.)  

![](https://velog.velcdn.com/images/tiger/post/eb26961e-7197-404c-a31c-6ccebbed69fd/image.png)
  
![](https://velog.velcdn.com/images/tiger/post/edc034d6-f4dc-448a-aaa9-0c358e8b4fe4/image.png)  

`ls -a`는 숨겨진 파일까지 보여주는 반면,`-a`옵션을 제거한 `ls`는 숨겨진 파일을 제외한 파일의 목록을 보여주므로 결과에 `.git`파일이 보이지 않음을 알 수 있다.  

### master branch
git파일을 생성하면 Git은 기본적으로 'master'라는 이름의 브랜치를 만든다. 이 새로운 저장소에 새로운 파일을 추가 한다거나 추가한 파일의 내용을 변경하여 그 내용을 commit하는 것은 모두 'master'라는 이름의 브랜치를 통해 처리할 수 있는 일이 된다.  
'master'가 아닌 또 다른 새로운 브랜치를 만들어 '앞으로 이 브랜치를 사용할거야!'라고 checkout하지 않는 이상, 이 때의 모든 작업은 'master'브랜치에서 이루어진다.  

## git workflow
`git init`을 하면 이제 해당 프로젝트의 파일들은 git의 작업환경에 올라가게 된다.  
git에는 총 3가지의 작업환경이 나뉘어져 있다.  
프로젝트의 파일들을 수정 생성하는 작업을 하는 **working directory**, 어느정도 작업하고 있다가 버전 히스토리에 저장할 준비가 되어있는 파일들을 옮겨놓는 **staging area**, 버전 히스토리를 가지고 있는 **.git repository(.git directory**)로 이루어진다.  
  
![](https://velog.velcdn.com/images/tiger/post/ab10f50b-2849-482e-85ea-ee3fb9bca025/image.png)
  
working directory를 엄밀히 말하면 git이 tracking을 하고있는 **tracked**, 새로만들어졌거나 아직 tracking이 되지 않은 **untracked**로 나눌 수 있다.  
깃이 tracking하고 있는 중인 tracked에서도 지금 이시점에 수정이 되었는지 여부에 따라 **unmodified , modified**로 나눌 수 있다. unmodified는 이전 버전과 비교했을 때 수정이 안 된 것이므로 modified만 staging area로 옮겨간다.  
  
이제 예시를 보며, 명령어와 함께 이해해보자.  
![](https://velog.velcdn.com/images/tiger/post/23f640c9-e05d-423b-a9ae-1c2212db07c2/image.png)  

echo 명령어를 이용해서 3개의 txt파일에 적어 저장하였다.  
방금 만들어진 새 파일들이기에 아직 untracked인 working directory에 위치한다.  

>마스터브랜치의 색상이 초록에서 노랑or 주황이 된 이유는?!  
working directory에 commit 되지 않은 변경 사항이 발생했음을 보여준다. 모든 변경사항들이 commit되면 master branch의 색상은 다시 초록색이 된다.  

### git add
![](https://velog.velcdn.com/images/tiger/post/7b2bb947-eff3-444a-a894-bcd3dd1c58f9/image.png)  
  
git이 tracking할 수 있도록 working directory에서 staging area에 올리려면 **add명령어**를 실행하면 된다.    
위와 같이 공백으로 파일 간의 간격을 주면 여러 파일을 한 번에 add 할 수 있다.   
결과적으로, a.txt, b.txt, c.txt모두 add되어 staging area에 올라갔음을 알 수 있다.  
  
자신의 위치부터 하위 폴더의 파일들까지 모두 올리려면 `git add .`명령어를 사용하면 된다.  
만약, txt파일만 업로드 하고 싶다면 `git add *.txt`를 사용하면 된다.  
  
staging area에 올라간 a.txt에 'ellie'를 추가를 하면 어떻게 될까?  
![](https://velog.velcdn.com/images/tiger/post/b651b357-6b2b-4243-8cb4-8821b8b65252/image.png)
  
동일한 파일이어도 추가된 'ellie'가 staging area의 a.txt에 존재하지 않기 때문에, 동일한 파일 일지라도 **수정이 이루어지면 다시 add, commit**을 해야한다.  
여기서 a.txt가 tracking이 되는 staging area에 있으므로 git status를 했을 때 modified: a.txt라고 뜨는 걸 볼 수 있다.  
![](https://velog.velcdn.com/images/tiger/post/5dd95f6a-a9c5-4c19-9c72-ede8bb9c19ba/image.png)
  
workflow에 보기좋게 표현하면 위와 같이 표현할 수 있다.  

### .gitignore 파일
로그나 빌드를 했을 때 빌드 안의 부수적인 파일들은 git에 포함하고 싶지 않을 수 있다. 이렇게 **숨기고 싶은 파일들이 존재할 때 사용하는 것이 .gitignore파일**이다.  
git과 github에 올리고 싶지 않은 파일(track 하고 싶지 않은 파일)들은 .gitignore이라는 파일에 추가하면 된다.  

log.log라는 파일을 숨기고 싶다고 하자.  

![](https://velog.velcdn.com/images/tiger/post/67b61f00-2c1f-4a41-af05-b7b4c1504d01/image.png)  

현재는 `git status`에 추가하지 않은 파일로 나타나는 것을 볼 수 있다.   
`echo *.log > .gitignore`로 .log파일들을 .gitignore 파일안에 넣었다.  
list를 보면 .gitignore가 생긴 것을 볼 수 있다.  
다시 git status를 하면 어떻게 나올까?  

![](https://velog.velcdn.com/images/tiger/post/18db3835-9a89-4620-82e6-0603c0ec2e2c/image.png)
  
log.log는 숨겨져 이제 보이지 않고 .gitignore만 보임을 알 수 있다.  

**폴더 파일 표현하기**  
`*.log` .log로 끝나는 파일  
`build/` build디렉토리 아래에 있는파일  
`build/.log`build 디렉토리 아래의 .log로 끝나는 파일  

#### git status
`git status`는 git에 의해 관리되는 파일들의 상태를 알려준다.
`git status -s`로 간단하게 보자.
![](https://velog.velcdn.com/images/tiger/post/362cf555-6fc9-4c77-9ae9-aa967b6a9bcc/image.png)

<span style="color:green">A</span>는 added가 되어 staging area에 있다는 뜻이다.
<span style="color:red">?</span>는 아직 트래킹이 되지 않는 working directory에 있다는 뜻이다.
여기서 c.txt 파일에 텍스트를 추가해보자.
![](https://velog.velcdn.com/images/tiger/post/65bafc23-3289-40af-9d01-9dd9cd22e6a3/image.png)
<span style="color:green">A</span><span style="color:red">M</span>은 stagingarea에 add가 되었고, working directory에서 modified되었다는 뜻이다.

### git commit  
staging area에 있는 a.txt 파일을 .git directory에 올리고 싶으면 어떤 명령어를 사용하면 될까?  
![](https://velog.velcdn.com/images/tiger/post/061722a5-a58d-4c4d-b54e-6b150c816fd0/image.png) 
  
**commit 명령어**를 이용해서 staging area에 있는 파일을 git diretory로 이동 즉, history에 저장할 수 있다.  
(이렇게 깃 디렉토리에 저장된 버전들은** checkout명령어**를 이용해 원하는 버전으로 돌아갈 수 있다.)  
  
![](https://velog.velcdn.com/images/tiger/post/08ae3e21-bc28-4860-a911-59752becab81/image.png)  
  
이렇게 저장된 깃히스토리는 나의 컴퓨터에만 보관된다. 다르게 말하면, 내 컴퓨터에 문제가 생기면 이 히스토리를 다 잃어버린다는 것이다.  
그래서 보통 .git directory는 **Github같은 서버에 push명령어를 이용해 업로드**를 하여, 히스토리를 안전하게 저장한다. ( 반대로 서버에서 local로 다운받고싶으면 **pull명령어**를 사용하면 된다.)  

여기서 각각의 버전들에 어떤 정보가 들어있는지 확인해보자.  
각각의 커밋들엔 스냅샷된 정보들을 기반으로 하여 고유한 해시코드가 부과된다. 해시코드를 이용하여 버전 정보에 대해 참조할 수 있다. commit에는 아이디 뿐만 아니라 버전관리 메시지와 누가 작성했는지, 날짜와 시간 같은 것도 작성된다.  
![](https://velog.velcdn.com/images/tiger/post/96d8e981-f690-471f-aca8-618a9c9f267b/image.png)
  

#### git add + commit 한꺼번에 하는 방법  
보통 `git add . `로 working directory에서 staging area로 옮기고, staging area에서 `git commit -m " ~ " `꼴을 사용하여 메세지를 남기며 commit을 하여 git directory로 옮긴다.  
만약 워킹 디렉토리에 있는 것이 모두 맘에 든다면 `git commit -am " ~ "`명령어를 사용하여 add와 commit을 한 번에 할 수 있다.  

### commit Tip  
깃디렉토리에 있는 우리의 커밋들은 작업들을 버전 별로 나누어 관리할 수 있는 유용한 history 창고이다. 그러므로 여러 변경들을 하나로 묶어 커밋하는 것은 의미가 없다.  

_Tip 1_  
의미가 있는 변경들을 의미를 알아볼 수 있는 메세지를 적어 커밋하는 것이 좋다. 보통 커밋의 메세지는 **현재형**으로 그리고 **동사형**으로 만드는 것이 좋다..(e.g. init, add, fix)
  
_Tip 2_  
![](https://velog.velcdn.com/images/tiger/post/8b72a6b7-6753-477c-9f12-62163354dfba/image.png)  

현업에서 많은 개발자들이 하는 실수이다.  
내가 `Fix crashing on login module` 즉,  crashing을 고쳤다는 메세지를 남기면 정말 그 code만 수정하여 commit 해야한다. '하는 김에 다른 버그들도 고치자'라는 마음으로 여러 코드를 건드리면 코드리뷰를 하거나, 히스토리를 바라볼 때 혼동이 온다.  

### git log v.s. git reflog
**git log**
프로젝트의 모든 커밋 내역을 보려면 `git log` 명령어를 입력하면 된다.
작업 디렉토리와 스테이징 영역을 제어하는 `git status`와 비교하여 `git log`는 커밋된 히스토리에서만 작동한다. 전반적으로 git log 명령은 리포지토리 기록을 살펴보고 프로젝트의 특정 버전을 찾는 데 유용하다.
git log 명령어를 통해 보여지는 log는 각 커밋에 대한 자세한 정보를 담고 있다. (작성자, hash 값, 날짜와 시간, 그리고 커밋 메세지)
`$ git log`
```
commit a7b4db3d6cfa0267712a3b6b6adbb1bc78f84abb (HEAD -> main, origin/main)
Author: seoyun <dhn04100@gmail.com>
Date:   Thu Jan 5 21:13:33 2023 +0900

    Git 폴더 이동

commit 2a2a4ad104445e46566a9996fca646408ad8df44
Author: seoyun <dhn04100@gmail.com>
Date:   Thu Jan 5 21:12:34 2023 +0900

    add Git workflow
```
만약 특정 커밋 시점의 코드로 되돌리고 싶다면, 아래 명령어를 사용할 수 있다.
`git checkout <commit-hash>` 또는 `git reset <commit-hash>`
`<commit-hash>` 를 git log 에서 보이는 커밋의 실제 hash 값으로 대체해줘야 한다.

다음처럼 --oneline으로 간략하게 볼 수도 있다.
`$ git log --oneline`
```
a7b4db3 (HEAD -> main, origin/main) Git 폴더 이동
2a2a4ad add Git workflow
```

특정 개수를 보고 싶다면 -n 플래그를 활용한다.
`$ git log -n 10` 최근 10개 커밋만 전시
깃을 그래프 형태로 깔끔하게 보고 싶을 때 아래와 같이 사용한다.
커밋의 전체적인 방향과 머지된 흐름도 파악할 수 있다.

`$ git log --oneline --decorate --graph`
커밋과 브랜치의 히스토리를 다양하고 쉽게 보여주는 Sourcetree나 GitHub Desktop 같은 GUI 툴을 사용하는 것을 추천한다.

**git show**
git show 명령어로 가장 최근 커밋의 정보를 확인할 수도 있다.
`$ git show`
```
commit a7b4db3d6cfa0267712a3b6b6adbb1bc78f84abb (HEAD -> main, origin/main)
Author: seoyun <dhn04100@gmail.com>
Date:   Thu Jan 5 21:13:33 2023 +0900

    Git 폴더 이동

diff --git a/CS/Git/workflow.md b/Git/workflow.md
similarity index 100%
rename from CS/Git/workflow.md
rename to Git/workflow.md
```
특정 커밋 정보를 확인하려면 git show 커밋 해시를 붙여주면 된다.
`$ git show c008c4785eeb14a395b4aa6cf9fa3b9e5896f5a4`
`$ git show HEAD^` # HEAD 포인터를 활용할 수도 있다.

**git reflog**
`reflog`는 엄격하게 로컬이며 리포지토리의 일부가 아니다. push, pull 또는 clone에도 포함되지 않는다. Git은 `git reflog` 도구를 사용하여 branch tip(마지막 커밋)에 대한 변경 사항을 추적한다.
git reflog 명령어를 통해 git reset, git rebase 명령어로 삭제된 커밋을 포함한 모든 커밋 히스토리를 볼 수 있다.
`$ git reflog`
```
a7b4db3 (HEAD -> main, origin/main) HEAD@{0}: commit: Git 폴더 이동
2a2a4ad HEAD@{1}: commit: add Git workflow
74f2bad HEAD@{2}: commit: fix 변수명 규칙
447a5a3 HEAD@{3}: commit: add JAVA ch2
a5b1039 HEAD@{4}: pull origin main: Fast-forward
```
git reflog는 이전 명령어(ex. `git reset --hard`)를 취소하고 싶을 때 유용하다.
git reset 명령어에 대한 설명은 아래에서 나오지만, 여기서 간략하게 git reflog를 사용하는 상황을 살펴본다.
만약 작업 중에 다음처럼 `git reset –hard`로 이전 커밋으로 돌아갔다고 가정한다.
`$ git reset 74f2bad --hard`
```
HEAD is now at 74f2bad fix 변수명 규칙
```
이때 일반적이라면 git reset 하기 전의 작업 내역으로 돌아갈 수 없지만, git reflog에는 이렇게 git reset 한 명령 내역까지 모두 남아있다.
`$ git reflog`

```
74f2bad HEAD@{0}: reset: moving to 74f2bad
a7b4db3 (HEAD -> main, origin/main) HEAD@{1}: commit: Git 폴더 이동
2a2a4ad HEAD@{2}: commit: add Git workflow
74f2bad HEAD@{3}: commit: fix 변수명 규칙
```
따라서 git reset --hard 한 명령을 취소하고 싶으면 (명령 이전으로 돌아가고 싶으면) git reflog 에서 해당 명령 직전의 커밋 해시 값을 참조하여 git reset --hard 하면 된다.
`$ git reset a7b4db3 --hard`
```
HEAD is now at a7b4db3 Git 폴더 이동
```

**git log v.s. git reflog**
1. `log`는 리포지토리 커밋 기록의 공개 레코드인 반면 `reflog`는 비공개라는 것이다. push, fetch 또는 pull 후에 git 로그는 git 저장소의 일부로 복제된다. 반면에 `git reflog`는 git 저장소에 포함되지 않는다.  
2. `git log`는 참조(헤드, 태그, 원격)에서 액세스할 수 있는 커밋 로그를 보여준다.
`git reflog`는 언제든지 리포지토리에서 참조되거나 참조된 모든 커밋 의 기록이다.