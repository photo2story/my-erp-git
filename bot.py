import asyncio
from discord.ext import commands
from my_flask_app.utils import search_projects_and_respond, is_valid_project, plot_project_status, plot_yearly_comparison

bot = commands.Bot(command_prefix='!')

@bot.command()
async def project(ctx, *, query: str = None):
    print(f'Command received: project analysis with query: {query}')
    if query is None:
        await ctx.send("프로젝트 코드를 입력해주세요.")
        return

    # 프로젝트 코드가 'C'로 시작하지 않는 경우 자동으로 검색 실행
    if not query.upper().startswith('C'):
        await ctx.send(f'"{query}" 관련 프로젝트를 검색합니다...')
        await search_projects_and_respond(ctx, query)
        return

    project_code = query.upper()
    if not is_valid_project(project_code):
        await ctx.send(f"유효하지 않은 프로젝트 코드입니다: {project_code}\n관련 프로젝트를 검색합니다...")
        await search_projects_and_respond(ctx, query)
        return

    await ctx.send(f'프로젝트 {project_code} 분석을 시작합니다.')
    try:
        # Results_plot.py의 plot_project_status 실행
        await plot_project_status(project_code)
        await asyncio.sleep(1)  # 각 분석 사이에 1초 대기
        
        # Results_plot_mpl.py의 plot_yearly_comparison 실행
        await plot_yearly_comparison(project_code)
        
        await ctx.send(f'프로젝트 {project_code} 분석이 완료되었습니다.')
    except Exception as e:
        error_message = f'프로젝트 {project_code} 분석 중 오류가 발생했습니다: {e}'
        await ctx.send(error_message)
        print(f'Error analyzing project {project_code}: {e}') 